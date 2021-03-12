#!/usr/bin/env python3

from simtk.openmm import XmlSerializer, app

from QUBEKit.parametrisation.base_parametrisation import Parametrisation


class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, input_file=None, fftype="CM1A/OPLS"):

        super().__init__(molecule, input_file, fftype)

        # self.check_xml()
        self.xml = self.input_file if self.input_file else f"{self.molecule.name}.xml"
        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = "XML input " + self.fftype

    def serialise_system(self):
        """Serialise the input XML system using openmm."""

        modeller = app.Modeller(
            self.molecule.to_openmm_topology(), self.molecule.openmm_coordinates()
        )

        forcefield = app.ForceField(self.xml)
        # Check for virtual sites
        try:
            system = forcefield.createSystem(
                modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None
            )
        except ValueError:
            print("Virtual sites were found in the xml file")
            modeller.addExtraParticles(forcefield)
            system = forcefield.createSystem(
                modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None
            )

        xml = XmlSerializer.serializeSystem(system)
        with open("serialised.xml", "w+") as out:
            out.write(xml)
