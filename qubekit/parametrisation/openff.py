from typing import TYPE_CHECKING, List, Optional

from pydantic import validator
from qcelemental.util import which_import
from simtk import unit
from simtk.openmm import System
from typing_extensions import Literal

from qubekit.parametrisation.base_parametrisation import Parametrisation
from qubekit.utils.helpers import hide_warnings

with hide_warnings():
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import (
        AngleHandler,
        BondHandler,
        ForceField,
        ProperTorsionHandler,
        get_available_force_fields,
        vdWHandler,
    )
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        UnassignedAngleParameterException,
        UnassignedBondParameterException,
        UnassignedMoleculeChargeException,
        UnassignedProperTorsionParameterException,
        UnassignedValenceParameterException,
    )

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class OpenFF(Parametrisation):
    """
    This class uses the OpenFFtoolkit to parametrise a molecule and load an OpenMM simulation.
    A serialised XML is then stored in the parameter dictionaries.
    """

    type: Literal["OpenFF"] = "OpenFF"
    force_field: str = "openff_unconstrained-2.0.0.offxml"

    def start_message(self, **kwargs) -> str:
        return f"Parametrising molecule with {self.force_field}."

    @classmethod
    def is_available(cls) -> bool:
        off = which_import(
            "openff.toolkit",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install openff-toolkit`.",
        )
        return off

    @classmethod
    def _improper_torsion_ordering(cls) -> str:
        return "smirnoff"

    @validator("force_field")
    def _check_forcefield(cls, force_field: str) -> str:
        """
        Make sure the supplied force field is valid.
        """

        openff_forcefields = [ff.lower() for ff in get_available_force_fields()]
        if force_field in openff_forcefields:
            return force_field.lower()
        else:
            raise ValueError(
                f"The force field {force_field} was not found by the openff-toolkit please chose from {openff_forcefields}."
            )

    def _build_system(
        self, molecule: "Ligand", input_files: Optional[List[str]] = None
    ) -> System:
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Create an openFF molecule from the rdkit molecule, we always have hydrogen by this point
        off_molecule = Molecule.from_rdkit(
            molecule.to_rdkit(),
            allow_undefined_stereo=True,
            hydrogens_are_explicit=True,
        )

        # Make the OpenMM system
        off_topology = off_molecule.to_topology()

        forcefield = ForceField(self.force_field)
        # we need to remove the constraints
        if "Constraints" in forcefield._parameter_handlers:
            del forcefield._parameter_handlers["Constraints"]

        try:
            # Parametrise the topology and create an OpenMM System.
            system = forcefield.create_openmm_system(
                off_topology,
            )
        except (
            UnassignedValenceParameterException,
            UnassignedBondParameterException,
            UnassignedProperTorsionParameterException,
            UnassignedAngleParameterException,
            UnassignedMoleculeChargeException,
            TypeError,
        ):
            # If this does not work then we have a molecule that is not in SMIRNOFF so we must add generics
            # and remove the charge handler to get some basic parameters for the moleucle
            new_bond = BondHandler.BondType(
                smirks="[*:1]~[*:2]",
                length="0 * angstrom",
                k="0.0 * angstrom**-2 * mole**-1 * kilocalorie",
            )
            new_angle = AngleHandler.AngleType(
                smirks="[*:1]~[*:2]~[*:3]",
                angle="0.0 * degree",
                k="0.0 * mole**-1 * radian**-2 * kilocalorie",
            )
            new_torsion = ProperTorsionHandler.ProperTorsionType(
                smirks="[*:1]~[*:2]~[*:3]~[*:4]",
                periodicity1="1",
                phase1="0.0 * degree",
                k1="0.0 * mole**-1 * kilocalorie",
                periodicity2="2",
                phase2="180.0 * degree",
                k2="0.0 * mole**-1 * kilocalorie",
                periodicity3="3",
                phase3="0.0 * degree",
                k3="0.0 * mole**-1 * kilocalorie",
                periodicity4="4",
                phase4="180.0 * degree",
                k4="0.0 * mole**-1 * kilocalorie",
                idivf1="1.0",
                idivf2="1.0",
                idivf3="1.0",
                idivf4="1.0",
            )
            new_vdw = vdWHandler.vdWType(
                smirks="[*:1]",
                epsilon=0 * unit.kilocalories_per_mole,
                sigma=0 * unit.angstroms,
            )
            new_generics = {
                "Bonds": new_bond,
                "Angles": new_angle,
                "ProperTorsions": new_torsion,
                "vdW": new_vdw,
            }
            for key, val in new_generics.items():
                forcefield.get_parameter_handler(key).parameters.insert(0, val)
            # This has to be removed as sqm will fail with unknown elements
            del forcefield._parameter_handlers["ToolkitAM1BCC"]
            del forcefield._parameter_handlers["Electrostatics"]
            # Parametrize the topology and create an OpenMM System.
            system = forcefield.create_openmm_system(off_topology)
        return system
