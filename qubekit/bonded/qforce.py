"""An interface to the QForce program for bonded parameter derivation via internal hessian fitting."""

import os
import shutil
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from pydantic import Field, PrivateAttr
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.parametrisation import Gromacs
from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class QForceHessianFitting(StageBase):
    """
    This class interfaces with qforce <https://github.com/selimsami/qforce> and allows users to do internal hessian
    fitting, this will fit all bonds, angles and rigid torsions and does not require an initial guess.
    """

    combination_rule: Literal["amber"] = Field(
        "amber",
        description="The nonbonded combination rule that should be used in QForce during fitting.",
    )
    charge_scaling: float = Field(
        1, description="The amount by which the charge should be scaled during fitting."
    )
    _combination_to_scaling: Dict[str, Dict[str, float]] = PrivateAttr(
        {"amber": {"lj": 0.5, "charge": 0.8333, "comb_rule": 2}}
    )
    _bonding_threshold: float = PrivateAttr(0.3)

    def start_message(self, **kwargs) -> str:
        return "Starting internal hessian fitting using QForce."

    def finish_message(self, **kwargs) -> str:
        return "Internal hessian fitting finshed, saving parameters."

    @classmethod
    def is_available(cls) -> bool:
        qforce = which_import(
            "qforce",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `pip install qforce`.",
        )
        return qforce

    def _get_combination_settings(self) -> Dict[str, float]:
        """
        For the given combination rules get the dictionary of settings.
        """
        return self._combination_to_scaling[self.combination_rule]

    def _generate_settings(self) -> StringIO:
        """
        Generate the qforce settings for the fitting.
        Note:
            The urey terms are set to off as we do not currently support them.
        """
        combination_settings = self._get_combination_settings()

        settings = f"""
        [ff]
        lennard_jones = ext
        ext_lj_fudge = {combination_settings["lj"]}
        ext_q_fudge = {combination_settings["charge"]}
        ext_comb_rule = {combination_settings["comb_rule"]}
        charge_scaling = {self.charge_scaling}
        [terms]
        urey = false
        """
        return StringIO(settings)

    @classmethod
    def _generate_atom_types(cls, molecule: "Ligand") -> Dict[str, Any]:
        """
        Generate the atom type groupings used to assign charges and lj values.
        Here we use the symmetry groups that were applied in symmetrisation to group the atoms which have the same charge and lj values.

        Returns:
            atom_types: {'lj_types': ['ha', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha'],
                        'atom_types': {'ca': [0.331521, 0.413379], 'ha': [0.262548, 0.0673624]},}
        """
        # now we want to gather the non-bonded terms per type
        atom_types = {}
        lj_types = []
        atom_symmetry_classes = molecule.atom_types
        for i in range(len(molecule.atoms)):
            # get the symmetry type
            lj_type = f"{molecule.atoms[i].atomic_symbol}{atom_symmetry_classes[i]}"
            lj_types.append(lj_type)
            # here we only add the type values once assuming symmetry has already been applied
            if lj_type not in atom_types:
                atom_value = molecule.NonbondedForce[i][1:]
                atom_types[lj_type] = atom_value

        return dict(lj_types=lj_types, atom_types=atom_types)

    @classmethod
    def _get_hessian_triangle(cls, hessian: np.array) -> np.array:
        """
        For the given hessian matrix reshape it to get a flat representation, of average values.
        """
        hessian_1d = []
        for i in range(len(hessian)):
            for j in range(i + 1):
                hessian_1d.append((hessian[i, j] + hessian[j, i]) / 2)
        hessian_1d = np.array(hessian_1d)
        return (
            hessian_1d
            * constants.HA_TO_KCAL_P_MOL
            * constants.KCAL_TO_KJ
            / (constants.BOHR_TO_ANGS ** 2)
        )

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        The main worker method of the class, this will take the ligand and fit the hessian using qforce and then return ligand with the optimized parameters.
        The ligand must contain some qm coordinates, a hessian matrix and wbo matrix.
        """
        from qforce import run_hessian_fitting_for_external

        qm_data = self._create_qm_data_dictionary(molecule=molecule)
        atom_types = self._generate_atom_types(molecule=molecule)
        qforce_settings = self._generate_settings()

        _ = run_hessian_fitting_for_external(
            molecule.name, qm_data=qm_data, ext_lj=atom_types, config=qforce_settings
        )
        # now we need to grab the top file
        output_folder = f"{molecule.name}_qforce"
        shutil.copy(os.path.join(output_folder, "gas.top"), "gas.top")
        shutil.copy(
            os.path.join(output_folder, f"{molecule.name}_qforce.itp"),
            f"{molecule.name}_qforce.itp",
        )
        # now get the parameters from the gromacs files
        gromacs = Gromacs()
        molecule = gromacs.run(
            molecule=molecule,
            input_files=[
                "gas.top",
            ],
        )
        # now we need to get the terms back into the molecule
        return molecule

    def _clean_wbo_matrix(self, wbo_matrix: np.array) -> np.array:
        """
        Take the wbo matrix and clean it to replicate orbital analysis.
        """
        # set all values too low to 0
        wbo_matrix[wbo_matrix < self._bonding_threshold] = 0
        return wbo_matrix

    def _create_qm_data_dictionary(self, molecule: "Ligand") -> Dict:
        """
        Create the QForce QM data dictionary required to run the hessian fitting.
        """
        qm_data = dict(
            n_atoms=len(molecule.atoms),
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            elements=np.array([atom.atomic_number for atom in molecule.atoms]),
            coords=molecule.coordinates,
            hessian=self._get_hessian_triangle(molecule.hessian),
            n_bonds=[atom.GetTotalDegree() for atom in molecule.to_rdkit().GetAtoms()],
            b_orders=self._clean_wbo_matrix(wbo_matrix=molecule.wbo),
            lone_e=[0 for _ in molecule.atoms],
            point_charges=[atom.aim.charge for atom in molecule.atoms],
        )
        return qm_data
