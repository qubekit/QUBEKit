"""
Classes that help with hessian fitting
"""
from QUBEKit.ligand import Ligand
from io import StringIO
from typing import Dict, Any
import numpy as np


class QforceHessianFitting:
    """
    This class interfaces with qforce <https://github.com/selimsami/qforce> and allows users to do internal hessian
    fitting, this will fit all bonds, angles and rigid torsions and does not require an initial guess.
    """

    def __init__(self, combination_rule: str = "amber", charge_scalling: float = 1):
        self.combination_rule = combination_rule.lower()
        self.charge_scalling = charge_scalling
        self._combination_to_scalling = {"amber": {"lj": 0.5, "charge": 0.8333, "comb_rule": 2},
                                         "opls": {"lj": 0.5, "charge": 0.5, "comb_rule": 3}}

    def _get_combination_settings(self) -> Dict[str, float]:
        """
        For the given combination rules get the dictionary of settings.
        """
        try:
            settings = self._combination_to_scalling[self.combination_rule]
            return settings
        except KeyError:
            raise KeyError(f"The combination rule {self.combination_rule} is not supported please chose from {list(self._combination_to_scalling.keys())}")

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
        charge_scalling = {self.charge_scalling}
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
            lj_type = molecule.atom_symmetry_classes[i]
            lj_types.append(lj_type)
            # here we only add the type values once assuming symmetry has already been applied
            #TODO should we enforce that symmetry is always applied?
            if lj_type not in atom_types:
                atom_value = molecule.NonbondedForce[i][1:]
                atom_types[lj_type] = atom_value

        return dict(lj_types=lj_types, atom_types=atom_types)

    @staticmethod
    def _get_upper_hessian_triangle(hessian: np.array) -> np.array:
        """
        For the given hessian matrix reshape it to get a flat representation of only the upper half.
        """
        flat_hess = np.triu(hessian).flatten()
        # remove 0 and return
        return flat_hess[flat_hess != 0]

    def fit_hessian(self, molecule: Ligand) -> Ligand:
        """
        The main worker method of the class, this will take the ligand and fit the hessian using qforce and then return ligand with the optimized parameters.
        The ligand must contain some qm coordinates, a hessian matrix and wbo matrix.
        """
        from qforce import run_hessian_fitting_for_external

        qm_data = self._create_qm_data_dictionary(molecule=molecule)
        atom_types = self._generate_atom_types(molecule=molecule)
        qforce_settings = self._generate_settings()

        terms = run_hessian_fitting_for_external(molecule.name, qm_data=qm_data, ext_lj=atom_types, config=qforce_settings)
        # now we need to get the terms back into the molecule
        print(terms)
        return molecule

    def _create_qm_data_dictionary(self, molecule: Ligand) -> Dict:
        """
        Create the QForce QM data dictionary required to run the hessian fitting.
        """
        qm_data = dict(n_atoms=len(molecule.atoms), charge=molecule.charge, multiplicity=molecule.multiplicity,
                       elements=np.array([atom.atomic_number for atom in molecule.atoms]), coords=molecule.coords['qm'],
                       hessian=self._get_upper_hessian_triangle(molecule.hessian), n_bonds=[len(atom.bonds) for atom in molecule.atoms],
                       b_orders=molecule.wbo, lone_e=np.zeros(len(molecule.atoms)), point_charges=[atom[0] for atom in molecule.NonbondedForce.values()])
        return qm_data
