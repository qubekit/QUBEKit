"""
A Charge derivation base class.
"""
import abc
from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from qubekit.charges.solvent_settings.base import SolventBase
from qubekit.utils.datastructures import LocalResource, QCOptions, StageBase, TDSettings

if TYPE_CHECKING:
    from qubekit.molecules import Ligand

T = TypeVar("T", bound=SolventBase)


class ChargeBase(StageBase):

    type: Literal["ChargeBase"] = "ChargeBase"
    solvent_settings: Optional[T] = Field(
        None,
        description="The settings used to calculate the electron density in implicit solvent.",
    )
    program: Literal["gaussian"] = Field(
        "gaussian",
        description="The name of the QM program to calculate the electron density.",
    )
    basis: Optional[str] = Field(
        None,
        description="The alternative basis set name, to specify a different "
        "one from that used for optimisations.",
    )
    method: Optional[str] = Field(
        None,
        description="The alternative method name, to specify a different one from that used for optimisations.",
    )
    td_settings: Optional[TDSettings] = Field(
        None,
        description="The alternative Time-Dependent calculation settings that should be used in the calculation.",
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

    def _get_qc_options(self) -> Optional[QCOptions]:
        """
        Extract a QCOptions model from the solvent settings.
        """
        if self.basis is not None and self.method is not None:
            return QCOptions(
                program=self.program,
                method=self.method,
                basis=self.basis,
                td_settings=self.td_settings,
            )
        return None

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        A template run method which makes sure symmetry is applied when requested.
        """
        # remove any extra sites as they will be invalidated by new charges
        molecule.extra_sites.clear_sites()
        local_options = kwargs.get("local_options")
        # use the alternative or if not the provided spec
        qc_spec = self._get_qc_options() or kwargs.get("qc_spec")
        molecule = self._run(molecule, local_options=local_options, qc_spec=qc_spec)
        # apply symmetry to the charge parameters
        molecule = self.apply_symmetrisation(molecule=molecule)
        # now store the reference values into the nonbonded force as a parameter
        for i in range(molecule.n_atoms):
            atom = molecule.atoms[i]
            molecule.NonbondedForce[(i,)].charge = atom.aim.charge

        molecule.fix_net_charge()

        return molecule

    @abc.abstractmethod
    def _gas_calculation_settings(self) -> Dict[str, Any]:
        """Build the gas phase settings dict for the calculation."""
        ...

    def _get_calculation_settings(self) -> Dict[str, Any]:
        """
        Build the calculation settings dict for the qcengine job.

        First we check for solvent keywords else we use the gas phase keywords.
        """
        if self.solvent_settings is not None:
            return self.solvent_settings.format_keywords()
        else:
            return self._gas_calculation_settings()

    @abc.abstractmethod
    def _run(
        self, molecule: "Ligand", local_options: LocalResource, qc_spec: QCOptions
    ) -> "Ligand":
        """
        The main method of the ChargeClass which should generate the charge and aim reference data and store it in the ligand.
        """
        ...
