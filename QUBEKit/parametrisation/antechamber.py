#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils.decorators import for_all_methods, timer_logger

import os
import shutil
import subprocess as sp
from tempfile import TemporaryDirectory

from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class AnteChamber(Parametrisation):
    """
    Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    def __init__(self, molecule, input_file=None, fftype='gaff'):

        super().__init__(molecule, input_file, fftype)

        self.antechamber_cmd()
        self.serialise_system()
        self.gather_parameters()
        self.get_symmetry()
        self.prmtop = None
        self.inpcrd = None
        self.molecule.parameter_engine = 'AnteChamber ' + self.fftype
        self.molecule.combination = self.combination

    def serialise_system(self):
        """Serialise the amber style files into an openmm object."""

        prmtop = app.AmberPrmtopFile(self.prmtop)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))

    def antechamber_cmd(self):
        """Method to run Antechamber, parmchk2 and tleap."""

        # file paths when moving in and out of temp locations
        cwd = os.getcwd()
        mol2 = os.path.abspath(f'{self.molecule.name}.mol2')
        pdb_path = os.path.abspath(f'{self.molecule.name}.pdb')
        frcmod_file = os.path.abspath(f'{self.molecule.name}.frcmod')
        prmtop_file = os.path.abspath(f'{self.molecule.name}.prmtop')
        inpcrd_file = os.path.abspath(f'{self.molecule.name}.inpcrd')
        ant_log = os.path.abspath('Antechamber.log')

        # Call Antechamber
        # Do this in a temp directory as it produces a lot of files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(pdb_path, 'in.pdb')

            # Call Antechamber
            cmd = f'antechamber -i in.pdb -fi pdb -o out.mol2 -fo mol2 -s 2 -at {self.fftype} -c bcc -nc {self.molecule.charge}'

            with open('ante_log.txt', 'w+') as log:
                sp.run(cmd, shell=True, stdout=log, stderr=log)

            # Ensure command worked
            try:
                # Copy the gaff mol2 and antechamber file back
                shutil.copy('out.mol2', mol2)
                shutil.copy('ante_log.txt', cwd)
            except FileNotFoundError:
                raise FileNotFoundError('Antechamber could not convert this file type; is it a valid pdb?')

            os.chdir(cwd)

        # Work in temp directory due to the amount of files made by antechamber
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(mol2, 'out.mol2')

            # Run parmchk
            with open('Antechamber.log', 'a') as log:
                sp.run(f'parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s {self.fftype}',
                       shell=True, stdout=log, stderr=log)

            # Ensure command worked
            if not os.path.exists('out.frcmod'):
                raise FileNotFoundError('out.frcmod not found parmchk2 failed!')

            # Now get the files back from the temp folder
            shutil.copy('out.mol2', mol2)
            shutil.copy('out.frcmod', frcmod_file)
            shutil.copy('Antechamber.log', ant_log)

        # Now we need to run tleap to get the prmtop and inpcrd files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(mol2, 'in.mol2')
            shutil.copy(frcmod_file, 'in.frcmod')
            shutil.copy(ant_log, 'Antechamber.log')

            # make tleap command file
            with open('tleap_commands', 'w+') as tleap:
                tleap.write("""source oldff/leaprc.ff99SB
                               source leaprc.gaff
                               LIG = loadmol2 in.mol2
                               check LIG
                               loadamberparams in.frcmod
                               saveamberparm LIG out.prmtop out.inpcrd
                               quit""")

            # Now run tleap
            with open('Antechamber.log', 'a') as log:
                sp.run('tleap -f tleap_commands', shell=True, stdout=log, stderr=log)

            # Check results present
            if not os.path.exists('out.prmtop') or not os.path.exists('out.inpcrd'):
                raise FileNotFoundError('Neither out.prmtop nor out.inpcrd found; tleap failed!')

            shutil.copy('Antechamber.log', ant_log)
            shutil.copy('out.prmtop', prmtop_file)
            shutil.copy('out.inpcrd', inpcrd_file)
            os.chdir(cwd)

        # Now give the file names to parametrisation method
        self.prmtop = f'{self.molecule.name}.prmtop'
        self.inpcrd = f'{self.molecule.name}.inpcrd'
