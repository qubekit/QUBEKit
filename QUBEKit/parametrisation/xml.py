#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils.decorators import for_all_methods, timer_logger

from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS'):

        super().__init__(molecule, input_file, fftype)

        # self.check_xml()
        self.serialise_system()
        self.gather_parameters()
        self.get_symmetry()
        self.molecule.parameter_engine = 'XML input ' + self.fftype
        self.molecule.combination = 'opls'

    def serialise_system(self):
        """Serialise the input XML system using openmm."""

        pdb = app.PDBFile(f'{self.molecule.name}.pdb')
        modeller = app.Modeller(pdb.topology, pdb.positions)

        if self.input_file:
            forcefield = app.ForceField(self.input_file)
        else:
            try:
                forcefield = app.ForceField(self.molecule.name + '.xml')
            except FileNotFoundError:
                raise FileNotFoundError('No .xml type file found.')

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        xml = XmlSerializer.serializeSystem(system)
        with open('serialised.xml', 'w+') as out:
            out.write(xml)
