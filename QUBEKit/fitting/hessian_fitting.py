"""
Classes that help with hessian fitting
"""
from QUBEKit.ligand import Ligand
from QUBEKit.utils import constants
from io import StringIO
from typing import Dict, Any
import numpy as np


class QforceHessianFitting:
    """
    This class interfaces with qforce <https://github.com/selimsami/qforce> and allows users to do internal hessian
    fitting, this will fit all bonds, angles and rigid torsions and does not require an initial guess.
    """

    def __init__(self, combination_rule: str = "amber", charge_scaling: float = 1, bonding_threshold: float = 0.3):
        self.combination_rule = combination_rule.lower()
        self.bonding_threshold = bonding_threshold
        self.charge_scaling = charge_scaling
        self._combination_to_scaling = {"amber": {"lj": 0.5, "charge": 0.8333, "comb_rule": 2},
                                         "opls": {"lj": 0.5, "charge": 0.5, "comb_rule": 3}}

    def _get_combination_settings(self) -> Dict[str, float]:
        """
        For the given combination rules get the dictionary of settings.
        """
        try:
            settings = self._combination_to_scaling[self.combination_rule]
            return settings
        except KeyError:
            raise KeyError(f"The combination rule {self.combination_rule} is not supported please chose from {list(self._combination_to_scaling.keys())}")

    def _generate_settings(self) -> StringIO:
        """
        Generate the qforce settings for the fitting.
        """
        combination_settings = self._get_combination_settings()

        settings = f"""
        [ff]
        lennard_jones = ext
        ext_lj_fudge = {combination_settings["lj"]}
        ext_q_fudge = {combination_settings["charge"]}
        ext_comb_rule = {combination_settings["comb_rule"]}
        charge_scaling = {self.charge_scaling}
        """
        return StringIO(settings)

    @staticmethod
    def _generate_atom_types(molecule: Ligand) -> Dict[str, Any]:
        """
        Generate the atom type groupings used to assign charges and lj values.
        Here we use the symmetry groups that were applied in symmetrisation to group the atoms which have the same charge and lj values.
        TODO would this be better in the ligand class? Could this help with file format conversions?

        returns
        -------
            atom_types: {'lj_types': ['ha', 'ca', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha', 'ca', 'ha'],
                        'atom_types': {'ca': [0.331521, 0.413379], 'ha': [0.262548, 0.0673624]},}
        """
        # make sure the molecule has symmetry classes
        if molecule.atom_symmetry_classes is None:
            from QUBEKit.engines import RDKit
            molecule.atom_symmetry_classes = RDKit.find_symmetry_classes(molecule.rdkit_mol)

        # now we want to gather the non-bonded terms per type
        atom_types = {}
        lj_types = []
        for i in range(len(molecule.atoms)):
            # get the symmetry type
            lj_type = f"{molecule.atoms[i].atomic_symbol}{molecule.atom_symmetry_classes[i]}"
            lj_types.append(lj_type)
            # here we only add the type values once assuming symmetry has already been applied
            #TODO should we enforce that symmetry is always applied?
            if lj_type not in atom_types:
                atom_value = molecule.NonbondedForce[i][1:]
                atom_types[lj_type] = atom_value

        return dict(lj_types=lj_types, atom_types=atom_types)

    @staticmethod
    def _get_hessian_triangle(hessian: np.array) -> np.array:
        """
        For the given hessian matrix reshape it to get a flat representation, of average values.
        """
        hessian_1d = []
        for i in range(len(hessian)):
            for j in range(i + 1):
                hessian_1d.append((hessian[i, j] + hessian[j, i]) / 2)
        hessian_1d = np.array(hessian_1d)
        return hessian_1d * constants.HA_TO_KCAL_P_MOL * constants.KCAL_TO_KJ / (constants.BOHR_TO_ANGS **2)

    def fit_hessian(self, molecule: Ligand) -> Ligand:
        """
        The main worker method of the class, this will take the ligand and fit the hessian using qforce and then return ligand with the optimized parameters.
        The ligand must contain some qm coordinates, a hessian matrix and wbo matrix.
        """
        from qforce import run_hessian_fitting_for_external
        from QUBEKit.parametrisation import Gromacs
        import os
        import shutil

        qm_data = self._create_qm_data_dictionary(molecule=molecule)
        atom_types = self._generate_atom_types(molecule=molecule)
        qforce_settings = self._generate_settings()

        _ = run_hessian_fitting_for_external(molecule.name, qm_data=qm_data, ext_lj=atom_types, config=qforce_settings)
        # now we need to grab the top file
        output_folder = f"{molecule.name}_qforce"
        shutil.copy(os.path.join(output_folder, "gas.top"), "gas.top")
        shutil.copy(os.path.join(output_folder, f"{molecule.name}_qforce.itp"), f"{molecule.name}_qforce.itp")
        # now get the parameters from the gromacs files
        gromacs = Gromacs()
        molecule = gromacs.parameterise_molecule(molecule=molecule, input_files=["gas.top", ])
        # now we need to get the terms back into the molecule
        return molecule

    def _clean_wbo_matrix(self, wbo_matrix: np.array) -> np.array:
        """
        Take the wbo matrix and clean it to replicate orbital anlysis.
        """
        # set all values too low to 0
        wbo_matrix[wbo_matrix < self.bonding_threshold] = 0
        return wbo_matrix

    def _create_qm_data_dictionary(self, molecule: Ligand) -> Dict:
        """
        Create the QForce QM data dictionary required to run the hessian fitting.
        """
        qm_data = dict(n_atoms=len(molecule.atoms), charge=molecule.charge, multiplicity=molecule.multiplicity,
                       elements=np.array([atom.atomic_number for atom in molecule.atoms]), coords=molecule.coords['qm'],
                       hessian=self._get_hessian_triangle(molecule.hessian), n_bonds=[atom.GetTotalDegree() for atom in molecule.rdkit_mol.GetAtoms()],
                       b_orders=self._clean_wbo_matrix(wbo_matrix=molecule.wbo), lone_e=[0 for _ in molecule.atoms], point_charges=[atom[0] for atom in molecule.NonbondedForce.values()])
        return qm_data
