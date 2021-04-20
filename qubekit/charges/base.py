"""
A Charge derivation base class.
"""
import abc
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from qubekit.engines import QCEngine
from qubekit.engines.base_engine import BaseEngine
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class ChargeBase(StageBase, BaseEngine):

    type: Literal["ChargeBase"] = "ChargeBase"
    apply_symmetry: bool = Field(
        True,
        description="Apply symmetry to the raw charge and volume values before assigning them.",
    )

    def build_engine(self) -> QCEngine:
        """
        Build a QCEngine instance with the settings we want to use.
        """
        engine = QCEngine(
            program=self.program,
            memory=self.memory,
            method=self.method,
            basis=self.basis,
            cores=self.cores,
            driver="energy",
        )
        return engine

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

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        A template run method which makes sure symmetry is applied when requested.
        """
        molecule = self._run(molecule, **kwargs)
        if self.apply_symmetry:
            molecule = self.apply_symmetrisation(molecule=molecule)
        # now store the reference values into the nonbonded force as a parameter
        for i in range(molecule.n_atoms):
            atom = molecule.atoms[i]
            molecule.NonbondedForce[(i,)].charge = atom.aim.charge
        return molecule

    @abc.abstractmethod
    def _run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        The main method of the ChargeClass which should generate the charge and aim reference data and store it in the ligand.
        """
        ...
