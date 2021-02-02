#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.ligand import Ligand

from typing import Optional, List, Tuple
import os
import shutil
import subprocess as sp
from tempfile import TemporaryDirectory

from simtk.openmm import System, app


class AnteChamber(Parametrisation):
    """
    Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    def __init__(self, fftype='gaff'):

        super().__init__(fftype)

    def build_system(self, molecule: Ligand, input_files: Optional[List[str]] = None) -> System:
        """Serialise the amber style files into an openmm object."""

        prmtop_file = self.make_prmtop(molecule=molecule)
        prmtop = app.AmberPrmtopFile(prmtop_file)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
        return system

    def make_prmtop(self, molecule: Ligand) -> Tuple[str, str]:
        """Method to run Antechamber, parmchk2 and tleap."""

        # file paths when moving in and out of temp locations
        cwd = os.getcwd()
        mol2 = os.path.abspath(f'{molecule.name}.mol2')
        pdb_path = os.path.abspath(f'{molecule.name}.pdb')
        frcmod_file = os.path.abspath(f'{molecule.name}.frcmod')
        prmtop_file = os.path.abspath(f'{molecule.name}.prmtop')
        inpcrd_file = os.path.abspath(f'{molecule.name}.inpcrd')
        ant_log = os.path.abspath('Antechamber.log')

        # Call Antechamber
        # Do this in a temp directory as it produces a lot of files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(pdb_path, 'in.pdb')

            # Call Antechamber
            cmd = f'antechamber -i in.pdb -fi pdb -o out.mol2 -fo mol2 -s 2 -at {self.fftype} -c bcc -nc {molecule.charge}'

            with open('ante_log.txt', 'w+') as log:
                sp.run(cmd, shell=True, stdout=log, stderr=log)

            # Ensure command worked
            try:
                # Copy the gaff mol2 and antechamber file back
                shutil.copy('ante_log.txt', cwd)
                shutil.copy('out.mol2', mol2)
            except FileNotFoundError:
                os.chdir(cwd)
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
        prmtop = f'{molecule.name}.prmtop'
        inpcrd = f'{molecule.name}.inpcrd'
        return (prmtop, inpcrd)
