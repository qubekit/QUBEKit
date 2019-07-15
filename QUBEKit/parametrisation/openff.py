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
        self.get_symmetry()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype
        self.molecule.combination = self.combination

    def serialise_system(self):
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Load the molecule using openforcefield
        pdb_file = app.PDBFile(f'{self.molecule.name}.pdb')

        # Now we need the connection info try using smiles string from rdkit
        rdkit = RDKit()
        molecule = Molecule.from_smiles(rdkit.get_smiles(f'{self.molecule.name}.pdb'))

        # Make the openMM system
        omm_topology = pdb_file.topology
        off_topology = Topology.from_openmm(omm_topology, unique_molecules=[molecule])

        # Load the smirnoff99Frosst force field.
        forcefield = ForceField('test_forcefields/smirnoff99Frosst.offxml')

        # Parametrize the topology and create an OpenMM System.
        system = forcefield.create_openmm_system(off_topology)

        # Serialise the OpenMM system into the xml file
        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))
