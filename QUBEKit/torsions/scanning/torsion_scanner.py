import copy
from typing import TYPE_CHECKING, List, Set, Tuple

from pydantic import Field
from qcelemental.util import which_import
from typing_extensions import Literal

from QUBEKit.engines import TorsionDriver
from QUBEKit.torsions.utils import AvoidedTorsion, TargetTorsion, find_heavy_torsion
from QUBEKit.utils.datastructures import StageBase, TorsionScan
from QUBEKit.utils.file_handling import folder_setup

if TYPE_CHECKING:
    from QUBEKit.molecules import Bond, Ligand


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
    avoided_torsions: Set[AvoidedTorsion] = Field(
        {
            AvoidedTorsion(smirks="[*:1]-[CH3:2]"),
            AvoidedTorsion(smirks="[*:1]-[NH2:2]"),
        },
        description="The list of torsion patterns that should be avoided.",
    )
    torsion_driver: TorsionDriver = Field(
        TorsionDriver(),
        description="The torsion drive engine used to compute the reference data.",
    )
    type: Literal["TorsionScan1D"] = "TorsionScan1D"

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
            molecule=drive_mol, torsion_scans=torsion_scans
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
        self, molecule: "Ligand", torsion_scans: List[TorsionScan]
    ) -> "Ligand":
        """
        Run the list of validated torsion drives.
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
                    molecule=molecule, dihedral_data=torsion_scan
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
        """
        torsion = AvoidedTorsion(smirks=smirks)
        self.avoided_torsions.add(torsion)

    def clear_avoided_torsions(self) -> None:
        """Remove all avoided torsions."""
        self.avoided_torsions = set()


# class TorsionScan:
#     """
#     This class will take a QUBEKit molecule object and perform a torsiondrive QM energy scan
#     for each selected rotatable dihedral.
#
#     inputs
#     ---------------
#     molecule                A QUBEKit Ligand instance
#     constraints_made        The name of the constraints file that should be used during the torsiondrive (pis4 only)
#
#     attributes
#     ---------------
#     qm_engine               An instance of the QM engine used for any calculations
#     native_opt              Chosen dynamically whether to use geometric or not (geometric is need to use constraints)
#     input_file               The name of the template file for tdrive, name depends on the qm_engine used
#     home                    The starting location of the job, helpful when scanning multiple angles.
#     """
#
#     def __init__(self, molecule: Ligand, constraints_made=None):
#
#         self.molecule = molecule
#         self.molecule.convergence = "GAU"
#         self.constraints_made = constraints_made
#
#         self.qm_engine = {"g09": Gaussian, "g16": Gaussian}.get(molecule.bonds_engine)(
#             molecule
#         )
#         self.native_opt = True
#
#         # Ensure geometric can only be used with psi4 so far
#         if molecule.geometric and molecule.bonds_engine == "psi4":
#             self.native_opt = False
#
#         self.input_file = None
#
#         self.home = os.getcwd()
#
#     def is_short_cc_bond(self, bond):
#         rdkit_mol = self.molecule.to_rdkit()
#         atom_0, atom_1 = [rdkit_mol.GetAtomWithIdx(atom) for atom in bond]
#
#         if atom_0.GetSymbol().upper() == "C" and atom_1.GetSymbol().upper() == "C":
#             if atom_0.GetDegree() == 3 and self.molecule.measure_bonds()[bond] < 1.42:
#                 return True
#         return False
#
#     def find_scan_order(self):
#         """
#         Function takes the molecule and displays the rotatable central bonds,
#         the user then enters the numbers of the torsions to be scanned (in the order they'll be scanned in).
#         The molecule can also be supplied with a scan order already, if coming from csv.
#         Else the user can supply a torsiondrive style QUBE_torsions.txt file we can extract the parameters from.
#         """
#
#         if self.molecule.scan_order:
#             return
#
#         # Get the rotatable dihedrals from the molecule
#         self.molecule.scan_order = []
#         rotatables_bonds = self.molecule.find_rotatable_bonds()
#         if rotatables_bonds is None:
#             return
#         rotatables_bonds = [bond.indices for bond in rotatables_bonds]
#
#         rotatable = set()
#         non_rotatable = set()
#
#         for dihedral_class in self.molecule.dihedral_types.values():
#             dihedral = dihedral_class[0]
#             bond = dihedral[1:3]
#             if bond in rotatables_bonds:
#                 rotatable.add(bond)
#             else:
#                 non_rotatable.add(dihedral)
#
#         for bond in rotatable:
#             dihedral = self.molecule.dihedrals[bond][0]
#             if self.is_short_cc_bond(bond):
#                 non_rotatable.add(dihedral)
#                 continue
#
#             self.molecule.dih_starts[dihedral] = -170
#             self.molecule.dih_ends[dihedral] = 180
#             rev_dihedral = tuple(reversed(dihedral))
#             self.molecule.dih_starts[rev_dihedral] = -170
#             self.molecule.dih_ends[rev_dihedral] = 180
#             self.molecule.scan_order.append(dihedral)
#             self.molecule.increments[dihedral] = 10
#             self.molecule.increments[rev_dihedral] = 10
#
#         if hasattr(self.molecule, "skip_nonrotatable"):
#             rdkit_mol = self.molecule.to_rdkit()
#             if self.molecule.skip_nonrotatable == "no":
#                 for dihedral in non_rotatable:
#                     bond = dihedral[1:3]
#                     restricted, half_restricted = False, False
#                     atom_0 = rdkit_mol.GetAtomWithIdx(bond[0])
#                     if (
#                         atom_0.IsInRingSize(3)
#                         or atom_0.IsInRingSize(4)
#                         or atom_0.IsInRingSize(5)
#                     ):
#                         restricted = True
#                     if not restricted:
#                         if self.is_short_cc_bond(bond):
#                             restricted = True
#
#                     current_val = self.molecule.measure_dihedrals()[dihedral]
#                     if restricted:
#                         atom_0 = rdkit_mol.GetAtomWithIdx(dihedral[0])
#                         atom_3 = rdkit_mol.GetAtomWithIdx(dihedral[3])
#                         half_restricted = not (atom_0.IsInRing() and atom_3.IsInRing())
#
#                     if restricted:
#                         # first or last atom are not part of a ring
#                         if half_restricted:
#                             self.molecule.dih_starts[dihedral] = int(current_val - 45.0)
#                             self.molecule.dih_ends[dihedral] = int(current_val + 45.0)
#                         # first and last part of a ring -> be careful
#                         else:
#                             self.molecule.dih_starts[dihedral] = int(current_val - 15.0)
#                             self.molecule.dih_ends[dihedral] = int(current_val + 15.0)
#
#                     else:
#                         if abs(current_val) <= 90:
#                             self.molecule.dih_starts[dihedral] = -100
#                             self.molecule.dih_ends[dihedral] = 100
#                         else:
#                             self.molecule.dih_starts[dihedral] = 80
#                             self.molecule.dih_ends[dihedral] = 280
#
#                     rev_dihedral = tuple(reversed(dihedral))
#                     self.molecule.dih_starts[rev_dihedral] = self.molecule.dih_starts[
#                         dihedral
#                     ]
#                     self.molecule.dih_ends[rev_dihedral] = self.molecule.dih_ends[
#                         dihedral
#                     ]
#                     self.molecule.scan_order.append(dihedral)
#                     self.molecule.increments[dihedral] = 5
#                     self.molecule.increments[rev_dihedral] = 5
#
#         self.molecule.scan_order.reverse()
#         print(f"scan order: {self.molecule.scan_order}")
#
#     def tdrive_scan_input(self, scan):
#         """Function takes the rotatable dihedrals requested and writes a scan input file for torsiondrive."""
#
#         scan_di = scan
#         if self.molecule.improper_torsions is None and len(scan) == 2:
#             scan_di = self.molecule.dihedrals[scan][0]
#
#         # Write the dihedrals.txt file for tdrive
#         with open("dihedrals.txt", "w+") as out:
#
#             out.write(
#                 "# dihedral definition by atom indices starting from 0\n# zero_based_numbering\n"
#                 "# i     j     k     l     "
#             )
#             out.write("(range_low)     (range_high)\n")
#             out.write(
#                 f"  {scan_di[0]}     {scan_di[1]}     {scan_di[2]}     {scan_di[3]}     "
#             )
#             out.write(
#                 f"{self.molecule.dih_starts[scan_di]}     {self.molecule.dih_ends[scan_di]}\n"
#             )
#
#         # Then write the template input file for tdrive in g09 or psi4 format
#         if self.native_opt:
#             self.qm_engine.generate_input(optimise=True, execute=False)
#             if self.qm_engine.__class__.__name__.lower() == "psi4":
#                 self.input_file = "input.dat"
#             else:
#                 self.input_file = f"gj_{self.molecule.name}.com"
#
#         else:
#             self.qm_engine.geo_gradient(execute=False, threads=True)
#             self.input_file = (
#                 f"{self.molecule.name}.{self.qm_engine.__class__.__name__.lower()}in"
#             )
#
#     def start_torsiondrive(self, scan):
#         """Start a torsiondrive either using psi4 or native gaussian09"""
#
#         # First set up the required files
#         self.tdrive_scan_input(scan)
#
#         # Now we need to run torsiondrive through the CLI
#         with open("tdrive.log", "w") as log:
#             # When we start the run write the options used here to be used during restarts
#             log.write(
#                 f"Theory used: {self.molecule.theory}   Basis used: {self.molecule.basis}\n"
#             )
#             log.flush()
#
#             if self.molecule.bonds_engine == "g09":
#                 tdrive_engine = "gaussian09"
#             elif self.molecule.bonds_engine == "g16":
#                 tdrive_engine = "gaussian16"
#             else:
#                 tdrive_engine = self.molecule.bonds_engine
#
#             span = self.molecule.dih_ends[scan] - self.molecule.dih_starts[scan]
#             step_size = 10
#             if span / step_size < 15:
#                 step_size = span / 35
#                 step_size = 5 * math.ceil(step_size / 5)
#
#             self.molecule.increments[scan] = step_size
#
#             cmd = (
#                 f"torsiondrive-launch -e {tdrive_engine} {self.input_file} dihedrals.txt -v -g {step_size}"
#                 f'{" --native_opt" if self.native_opt else ""}'
#             )
#
#             if not os.path.exists("qdata.txt"):
#                 sp.run(cmd, shell=True, stdout=log, check=True, stderr=log, bufsize=0)
#
#         if self.molecule.bonds_engine in ["g09", "g16"]:
#             Gaussian.cleanup()
#
#         # Gather the results
#         try:
#             self.molecule.read_tdrive(scan)
#         except FileNotFoundError as exc:
#             if not self.molecule.tdrive_parallel:
#                 raise TorsionDriveFailed(
#                     "Torsiondrive output qdata.txt missing; job did not execute or finish properly"
#                 ) from exc
#
#     def collect_scan(self):
#         """
#         Collect the results of a torsiondrive scan that has not been done using QUBEKit.
#         :return: The energies and coordinates into the molecule
#         """
#
#         for scan in self.molecule.scan_order:
#             name = self._make_folder_name(scan)
#             os.chdir(os.path.join(self.home, os.path.join(name, "QM_torsiondrive")))
#             self.molecule.read_tdrive(scan[1:3])
#
#     def _make_folder_name(self, scan):
#         return f"SCAN_{scan[0]}_{scan[1]}_{scan[2]}_{scan[3]}"
#
#     def check_run_history(self, scan):
#         """
#         Check the contents of a scan folder to see if we should continue the run;
#         if the settings (theory and basis) are the same then continue
#         :return: If we should continue the run
#         """
#
#         name = self._make_folder_name(scan)
#         # Try and open the tdrive.log file to check the old running options
#         try:
#             file_path = os.path.join(os.getcwd(), name, "QM_torsiondrive", "tdrive.log")
#             with open(file_path) as t_log:
#                 header = t_log.readline().split("Basis used:")
#
#             # This is needed for the case of a split theory e.g. EmpiricalDispersion=GD3BJ B3LYP
#             basis = header[1].strip()
#             theory = header[0].split("Theory used:")[1].strip()
#             return self.molecule.theory == theory and self.molecule.basis == basis
#
#         except FileNotFoundError:
#             return False
#
#     def scan(self):
#         """Makes a folder and writes a new a dihedral input file for each scan and runs the scan."""
#
#         for scan in self.molecule.scan_order:
#             name = self._make_folder_name(scan)
#             try:
#                 os.mkdir(name)
#             except FileExistsError:
#                 # If there is a run in the folder, check if we are continuing an old run by matching the settings
#                 con_scan = self.check_run_history(scan)
#                 if not con_scan:
#                     print(f"{name} folder present backing up folder to {name}_tmp")
#                     # Remove old backups
#                     try:
#                         rmtree(f"{name}_tmp")
#                     except FileNotFoundError:
#                         pass
#
#                     os.system(f"mv {name} {name}_tmp")
#                     os.mkdir(name)
#
#             os.chdir(name)
#
#             make_and_change_into("QM_torsiondrive")
#
#             # Start the torsion drive if psi4 else run native separate optimisations using g09
#             self.start_torsiondrive(scan)
#             # Get the scan results and load into the molecule
#             os.chdir(self.home)
