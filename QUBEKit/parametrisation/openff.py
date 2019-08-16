#!/usr/bin/env python3

from QUBEKit.engines import RDKit
from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils.decorators import for_all_methods, timer_logger

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField

from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class OpenFF(Parametrisation):
    """
    This class uses the openFFtoolkit 2 to parametrise a molecule and load an OpenMM simulation.
    A serialised XML is then stored in the parameter dictionaries.
    """

    def __init__(self, molecule, input_file=None, fftype='frost'):
        super().__init__(molecule, input_file, fftype)

        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype
        self.molecule.combination = self.combination

    def serialise_system(self):
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Create an openFF molecule from the rdkit molecule
        try:
            off_molecule = Molecule.from_rdkit(self.molecule.rdkit_mol, allow_undefined_stereo=True)
        except AttributeError:
            raise AttributeError('An rdkit molecule object is required but could not be generated from the input file.')

        # Make the openMM system
        off_topology = off_molecule.to_topology()

        # Load the smirnoff99Frosst force field.
        forcefield = ForceField('test_forcefields/smirnoff99Frosst.offxml')

        # Parametrize the topology and create an OpenMM System.
        system = forcefield.create_openmm_system(off_topology)

        # Serialise the OpenMM system into the xml file
        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))
