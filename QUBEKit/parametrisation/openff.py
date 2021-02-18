#!/usr/bin/env python3

from simtk import unit
from simtk.openmm import XmlSerializer

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils.helpers import hide_warnings

with hide_warnings():
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import (
        AngleHandler,
        BondHandler,
        ForceField,
        ProperTorsionHandler,
        vdWHandler,
    )
    from openff.toolkit.typing.engines.smirnoff.parameters import (
        UnassignedAngleParameterException,
        UnassignedBondParameterException,
        UnassignedMoleculeChargeException,
        UnassignedProperTorsionParameterException,
        UnassignedValenceParameterException,
    )


class OpenFF(Parametrisation):
    """
    This class uses the OpenFFtoolkit 2 to parametrise a molecule and load an OpenMM simulation.
    A serialised XML is then stored in the parameter dictionaries.
    """

    def __init__(self, molecule, input_file=None, fftype="frost"):

        super().__init__(molecule, input_file, fftype)

        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = f"OpenFF_{self.fftype}"

    def serialise_system(self):
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Create an openFF molecule from the rdkit molecule
        off_molecule = Molecule.from_rdkit(
            self.molecule.rdkit_mol, allow_undefined_stereo=True
        )

        # Make the OpenMM system
        off_topology = off_molecule.to_topology()

        forcefield = ForceField("openff_unconstrained-1.3.0.offxml")

        try:
            # Parametrise the topology and create an OpenMM System.
            system = forcefield.create_openmm_system(off_topology)
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
            # This will tag the molecule so run.py knows that generics have been used.
            self.fftype = "generics"
        # Serialise the OpenMM system into the xml file
        with open("serialised.xml", "w+") as out:
            out.write(XmlSerializer.serializeSystem(system))
