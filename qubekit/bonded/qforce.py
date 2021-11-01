"""An interface to the QForce program for bonded parameter derivation via internal hessian fitting."""
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from pydantic import Field, PrivateAttr
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase
from qubekit.utils.helpers import check_improper_torsion, check_proper_torsion

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class QForceHessianFitting(StageBase):
    """
    This class interfaces with qforce <https://github.com/selimsami/qforce> and allows users to do internal hessian
    fitting, this will fit all bonds, angles and rigid torsions and does not require an initial guess.
    """

    type: Literal["QForceHessianFitting"] = "QForceHessianFitting"
    combination_rule: Literal["amber"] = Field(
        "amber",
        description="The nonbonded combination rule that should be used in QForce during fitting.",
    )
    charge_scaling: float = Field(
        1, description="The amount by which the charge should be scaled during fitting."
    )
    use_urey_bradley: bool = Field(
        False,
        description="If Urey-Bradley coupling terms should also be fit using QForce.",
    )

    _combination_to_scaling: Dict[str, Dict[str, float]] = PrivateAttr(
        {"amber": {"lj": 0.5, "charge": 0.8333, "comb_rule": 2}}
    )

    def start_message(self, **kwargs) -> str:
        return "Starting internal hessian fitting using QForce."

    def finish_message(self, **kwargs) -> str:
        return "Internal hessian fitting finished, saving parameters."

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
        Generate the QForce settings for the fitting.
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
        ext_h_cap = H0
        charge_scaling = {self.charge_scaling}
        [terms]
        urey = {self.use_urey_bradley}
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
                atom_value = molecule.NonbondedForce[(i,)]
                atom_types[lj_type] = (atom_value.sigma, atom_value.epsilon)

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

    @classmethod
    def _save_parameters(cls, molecule: "Ligand", qforce_terms) -> None:
        """Update the Ligand with the final parameters from the QForce hessian fitting."""
        from qforce.forces import convert_to_inversion_rb

        # get all of the flexible dihedrals which qforce has not fit yet and keep our parent parameters
        center_bonds = []
        for flexible_torsion in qforce_terms["dihedral"]["flexible"]:
            center_bonds.append(tuple(flexible_torsion.atomids)[1:3])
        flexible_torsions = []
        for torsion in molecule.TorsionForce:
            if (
                torsion.atoms[1:3] in center_bonds
                or tuple(reversed(torsion.atoms[1:3])) in center_bonds
            ):
                flexible_torsions.append(torsion)

        # remove any starting parameters
        molecule.BondForce.clear_parameters()
        molecule.AngleForce.clear_parameters()
        molecule.TorsionForce.clear_parameters()
        # qforce only adds impropers when there are no rigid terms so remove any initial terms
        molecule.ImproperTorsionForce.clear_parameters()
        molecule.RBTorsionForce.clear_parameters()
        # this just means the improper should be constructed in the order defined by the forcefield
        molecule.TorsionForce.ordering = "charmm"

        for bond in qforce_terms["bond"]:
            # we only want terms which are non zero
            if round(bond.fconst * 100, ndigits=3) != 0:
                bond_data = {
                    "length": bond.equ * constants.ANGS_TO_NM,
                    "k": bond.fconst * 100,
                }
                molecule.BondForce.create_parameter(
                    atoms=tuple(bond.atomids), **bond_data
                )
        for angle in qforce_terms["angle"]:
            if round(angle.fconst, ndigits=3) != 0:
                angle_data = {"angle": angle.equ, "k": angle.fconst}
                molecule.AngleForce.create_parameter(tuple(angle.atomids), **angle_data)
        if "urey" in qforce_terms:
            # grab urey angle terms and make a bond for the terminal atoms
            for urey in qforce_terms["urey"]:
                if round(urey.fconst * 100, ndigits=3) != 0:
                    urey_data = {
                        "length": urey.equ * constants.ANGS_TO_NM,
                        "k": urey.fconst * 100,
                    }
                    atoms = urey.atomids
                    # make sure to grab the terminal atoms when making the bond term
                    molecule.BondForce.create_parameter(
                        atoms=tuple([atoms[0], atoms[2]]), **urey_data
                    )
        for dihedral in qforce_terms["dihedral"]["rigid"]:
            if round(dihedral.fconst / 4, ndigits=3) != 0:
                dihedral_data = {
                    "k2": dihedral.fconst / 4,
                }
                molecule.TorsionForce.create_parameter(
                    atoms=tuple(dihedral.atomids), **dihedral_data
                )
        for improper in qforce_terms["dihedral"]["improper"]:
            if round(improper.fconst / 4, ndigits=3) != 0:
                molecule.ImproperTorsionForce.create_parameter(
                    atoms=tuple(improper.atomids), k2=improper.fconst / 4
                )
        for inversion in qforce_terms["dihedral"]["inversion"]:
            # use the RB torsion type to model the inversion dihedrals
            # get the parameters in RB form
            c0, c1, c2 = convert_to_inversion_rb(inversion.fconst, inversion.equ)
            c0, c1, c2 = (
                round(c0, ndigits=3),
                round(c1, ndigits=3),
                round(c2, ndigits=3),
            )
            inversion_data = {"c0": c0, "c1": c1, "c2": c2}
            if check_proper_torsion(
                torsion=tuple(inversion.atomids), molecule=molecule
            ):
                molecule.RBTorsionForce.create_parameter(
                    atoms=tuple(inversion.atomids), **inversion_data
                )
            else:
                improper = check_improper_torsion(
                    improper=tuple(inversion.atomids), molecule=molecule
                )
                molecule.ImproperRBTorsionForce.create_parameter(
                    atoms=improper, **inversion_data
                )

        # now add back the flexible torsions
        molecule.TorsionForce.parameters.extend(flexible_torsions)

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        The main worker method of the class, this will take the ligand and fit the hessian using qforce and then return ligand with the optimized parameters.
        The ligand must contain some qm coordinates, a hessian matrix and wbo matrix.
        """
        from qforce import run_hessian_fitting_for_external

        qm_data = self._create_qm_data_dictionary(molecule=molecule)
        atom_types = self._generate_atom_types(molecule=molecule)
        qforce_settings = self._generate_settings()
        result = run_hessian_fitting_for_external(
            molecule.name, qm_data=qm_data, ext_lj=atom_types, config=qforce_settings
        )
        # now we need to get the terms back into the molecule
        self._save_parameters(molecule=molecule, qforce_terms=result)
        return molecule

    def _create_qm_data_dictionary(self, molecule: "Ligand") -> Dict:
        """
        Create the QForce QM data dictionary required to run the hessian fitting.
        """
        # try and get the point charges from AIM first, as virtual sites are not supported in qforce
        # we need to use the raw aim data
        point_charges = [atom.aim.charge for atom in molecule.atoms]
        if not any(point_charges):
            # if we have no aim data there should be no sites so use the forcefield data
            point_charges = [
                float(molecule.NonbondedForce[(i,)].charge)
                for i in range(molecule.n_atoms)
            ]
        qm_data = dict(
            n_atoms=molecule.n_atoms,
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            elements=np.array([atom.atomic_number for atom in molecule.atoms]),
            coords=molecule.coordinates,
            hessian=self._get_hessian_triangle(molecule.hessian),
            n_bonds=[atom.GetTotalValence() for atom in molecule.to_rdkit().GetAtoms()],
            b_orders=molecule.wbo,
            lone_e=[0 for _ in molecule.atoms],
            point_charges=point_charges,
        )
        return qm_data
