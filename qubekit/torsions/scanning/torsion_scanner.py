import copy
from typing import TYPE_CHECKING, List, Tuple

from pydantic import Field
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.engines import TorsionDriver
from qubekit.torsions.utils import AvoidedTorsion, TargetTorsion, find_heavy_torsion
from qubekit.utils.datastructures import (
    LocalResource,
    QCOptions,
    StageBase,
    TorsionScan,
)
from qubekit.utils.file_handling import folder_setup

if TYPE_CHECKING:
    from qubekit.molecules import Bond, Ligand


class TorsionScan1D(StageBase):
    """
    A 1D torsion scanner.

    Note:
        By default this will scan all rotatable bonds not involving methyl or amine terminal groups.
    """

    special_torsions: List[TargetTorsion] = Field(
        [],
        description="A list of special target torsions to be scanned and their scan range.",
    )
    default_scan_range: Tuple[int, int] = Field(
        (-165, 180),
        description="The default scan range for any torsions not covered by a special rule.",
    )
    avoided_torsions: List[AvoidedTorsion] = Field(
        [
            AvoidedTorsion(smirks="[*:1]-[CH3:2]"),
            AvoidedTorsion(smirks="[*:1]-[NH2:2]"),
        ],
        description="The list of torsion patterns that should be avoided.",
    )
    torsion_driver: TorsionDriver = Field(
        TorsionDriver(),
        description="The torsion drive engine used to compute the reference data.",
    )
    type: Literal["TorsionScan1D"] = "TorsionScan1D"

    def start_message(self, **kwargs) -> str:
        return f"Performing QM-constrained optimisation with Torsiondrive and {kwargs['qc_spec'].program}"

    def finish_message(self, **kwargs) -> str:
        return "Torsiondrive finished and QM results saved."

    @classmethod
    def is_available(cls) -> bool:
        """
        Make sure geometric and torsiondrive are installed.
        """
        geo = which_import(
            "geometric",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install geometric -c conda-forge`.",
        )
        tdrive = which_import(
            "torsiondrive",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install torsiondrive -c conda-forge`.",
        )
        engine = which_import(
            "qcengine",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install qcengine -c conda-forge`.",
        )
        return geo and tdrive and engine

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Run any possible torsiondrives for this molecule given the list of allowed and disallowed torsion patterns.

        Note:
            This function just validates the molecule and builds a list of torsions to scan before passing to the main method.
        Important:
            We work with a copy of the input molecule as we change the coordinates throughout the calculation.
        """
        # work with a copy as we change coordinates from the qm a lot!
        drive_mol = copy.deepcopy(molecule)
        # first find all rotatable bonds, while removing the unwanted scans
        bonds = drive_mol.find_rotatable_bonds(
            smirks_to_remove=[torsion.smirks for torsion in self.avoided_torsions]
        )
        if bonds is None:
            print("No rotatable bonds found to scan!")
            return molecule

        torsion_scans = []
        for bond in bonds:
            # get the scan range and a torsion for the bond
            torsion = find_heavy_torsion(molecule=drive_mol, bond=bond)
            scan_range = self._get_scan_range(molecule=drive_mol, bond=bond)
            torsion_scans.append(TorsionScan(torsion=torsion, scan_range=scan_range))

        result_mol = self._run_torsion_drives(
            molecule=drive_mol,
            torsion_scans=torsion_scans,
            qc_spec=kwargs["qc_spec"],
            local_options=kwargs["local_options"],
        )
        # make sure we preserve the input coords
        result_mol.coordinates = copy.deepcopy(molecule.coordinates)
        # make sure we have all of the scans we expect
        assert len(result_mol.qm_scans) == len(bonds)
        return result_mol

    def _get_scan_range(self, molecule: "Ligand", bond: "Bond") -> Tuple[int, int]:
        """
        Get the scan range for the target bond based on the allowed substructure list.

        Note:
            We loop over the list of targets checking each meaning that the last match in the list will be applied to substructure.
            So generic matches should be placed at the start of the list with more specific ones at the end.
        """
        scan_range = self.default_scan_range
        for target_torsion in self.special_torsions:
            matches = molecule.get_smarts_matches(smirks=target_torsion.smirks)
            for match in matches:
                if len(match) == 4:
                    atoms = match[1:3]
                else:
                    atoms = match

                if set(atoms) == set(bond.indices):
                    scan_range = target_torsion.scan_range

        return scan_range

    def _run_torsion_drives(
        self,
        molecule: "Ligand",
        torsion_scans: List[TorsionScan],
        qc_spec: QCOptions,
        local_options: LocalResource,
    ) -> "Ligand":
        """
        Run the list of validated torsion drives.

        Note:
            We do not change the initial coordinates passed at this point.

        Args:
            molecule: The molecule to be scanned.
            torsion_scans: A list of TorsionScan jobs to perform detailing the dihedral and the scan range.

        Returns:
            The updated molecule object with the scan results.
        """
        for torsion_scan in torsion_scans:
            # make a folder and move into to run the calculation
            folder = "SCAN_"
            folder += "_".join([str(t) for t in torsion_scan.torsion])
            with folder_setup(folder):
                print(
                    f"Running scan for dihedral: {torsion_scan.torsion} with range: {torsion_scan.scan_range}"
                )
                result_mol = self.torsion_driver.run_torsiondrive(
                    molecule=molecule,
                    dihedral_data=torsion_scan,
                    qc_spec=qc_spec,
                    local_options=local_options,
                )

        return result_mol

    def add_special_torsion(
        self, smirks: str, scan_range: Tuple[int, int] = (-165, 180)
    ) -> None:
        """
        Add a new allowed torsion scan.

        Args:
            smirks: The smirks pattern that should be used to identify the torsion.
            scan_range: The scan range for this type of dihedral.
        """
        target = TargetTorsion(smirks=smirks, scan_range=scan_range)
        self.special_torsions.append(target)

    def clear_special_torsions(self) -> None:
        """Remove all allowed torsion scans."""
        self.special_torsions = []

    def add_avoided_torsion(self, smirks: str) -> None:
        """
        Add a new torsion pattern to avoid scanning.

        Args:
            smirks: The valid smirks pattern which describes a torsion not to be scanned.
        """
        torsion = AvoidedTorsion(smirks=smirks)
        self.avoided_torsions.append(torsion)

    def clear_avoided_torsions(self) -> None:
        """Remove all avoided torsions."""
        self.avoided_torsions = []
