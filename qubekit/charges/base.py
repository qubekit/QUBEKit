"""
A Charge derivation base class.
"""
import abc
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from qubekit.charges.solvent_settings.base import SolventBase
from qubekit.utils.datastructures import LocalResource, QCOptions, StageBase

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class ChargeBase(StageBase):

    type: Literal["ChargeBase"] = "ChargeBase"
    solvent_settings: SolventBase = Field(
        ...,
        description="The settings used to calculate the electron density in implicit solvent.",
    )

    def finish_message(self, **kwargs) -> str:
        return "Charges calculated and AIM reference data stored."

    @classmethod
    def apply_symmetrisation(cls, molecule: "Ligand") -> "Ligand":
        """
        Apply symmetry to the aim charge and volume reference data in a molecule.
        """
        atom_types = {}
        for atom_index, cip_type in molecule.atom_types.items():
            atom_types.setdefault(cip_type, []).append(atom_index)

        for sym_set in atom_types.values():
            mean_charge = np.array(
                [molecule.atoms[ind].aim.charge for ind in sym_set]
            ).mean()

            mean_volume = np.array(
                [molecule.atoms[ind].aim.volume for ind in sym_set]
            ).mean()

            for atom_index in sym_set:
                molecule.atoms[atom_index].aim.charge = mean_charge
                molecule.atoms[atom_index].aim.volume = mean_volume

        return molecule

    def _get_qc_options(self) -> QCOptions:
        """
        Extract a QCOptions model from the solvent settings.
        """
        return QCOptions(
            program=self.solvent_settings.program,
            method=self.solvent_settings.method,
            basis=self.solvent_settings.basis,
        )

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        A template run method which makes sure symmetry is applied when requested.
        """
        local_options = kwargs.get("local_options")
        molecule = self._run(molecule, local_options=local_options)
        # apply symmetry to the charge parameters
        molecule = self.apply_symmetrisation(molecule=molecule)
        # now store the reference values into the nonbonded force as a parameter
        for i in range(molecule.n_atoms):
            atom = molecule.atoms[i]
            molecule.NonbondedForce[(i,)].charge = atom.aim.charge
        return molecule

    @abc.abstractmethod
    def _run(self, molecule: "Ligand", local_options: LocalResource) -> "Ligand":
        """
        The main method of the ChargeClass which should generate the charge and aim reference data and store it in the ligand.
        """
        ...
