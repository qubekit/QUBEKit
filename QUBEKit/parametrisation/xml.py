#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.ligand import Ligand

from typing import Optional, List
from simtk.openmm import app, System


class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, fftype='CM1A/OPLS'):
        super().__init__(fftype)

    def build_system(self, molecule: Ligand, input_files: Optional[List[str]] = None) -> System:
        """Serialise the input XML system using openmm."""

        # find the xml file
        xml = None
        if input_files is not None:
            for file in input_files:
                if file.endswith(".xml"):
                    xml = file
                    break
        # if we did not find one guess the name
        xml = xml or f"{molecule.name}.xml"

        pdb = app.PDBFile(f'{molecule.name}.pdb')
        modeller = app.Modeller(pdb.topology, pdb.positions)

        forcefield = app.ForceField(xml)
        # Check for virtual sites
        try:
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        except ValueError:
            print('Virtual sites were found in the xml file')
            modeller.addExtraParticles(forcefield)
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        return system
