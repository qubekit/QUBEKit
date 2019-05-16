from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.parametrisation.parameter_engines import Parametrisation
from QUBEKit.engines import RDKit

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField

from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class OpenFF(Parametrisation):
    """
    This class uses the openFFtoolkit 2 to parametrise a molecule and load an OpenMM simulation.
    A serialised XML is then stored in the parameter dictionaries.
    """

    def __init__(self, molecule, input_file=None, fftype='frost', mol2_file=None):
        super().__init__(molecule, input_file, fftype)

        self.get_gaff_types(mol2_file)
        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype
        self.molecule.combination = self.combination

    def serialise_system(self):
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Load the molecule using openforcefield
        pdb_file = app.PDBFile(self.molecule.filename)

        # Now we need the connection info try using smiles string from rdkit
        molecule = Molecule.from_smiles(RDKit.get_smiles(self.molecule.filename))

        # Make the openMM system
        omm_topology = pdb_file.topology
        off_topology = Topology.from_openmm(omm_topology, unique_molecules=[molecule])

        # Load the smirnof99Frosst force field.
        forcefield = ForceField('smirnoff99Frosst.offxml')

        # Parametrize the topology and create an OpenMM System.
        system = forcefield.create_openmm_system(off_topology)

        # Serialise the OpenMM system into the xml file
        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))
