#!/usr/bin/env python3

from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.engines.base_engine import Engines
from QUBEKit.helpers import check_symmetry

import subprocess as sp

import numpy as np


@for_all_methods(timer_logger)
class Gaussian(Engines):
    """
    Writes and executes input files for Gaussian09.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule):

        super().__init__(molecule)

        self.functional_dict = {'pbe': 'PBEPBE', 'wb97x-d': 'wB97XD'}
        self.molecule.theory = self.functional_dict.get(self.molecule.theory, self.molecule.theory)

        self.convergence_dict = {'GAU': '',
                                 'GAU_TIGHT': 'tight',
                                 'GAU_LOOSE': 'loose',
                                 'GAU_VERYTIGHT': 'verytight'}

    def generate_input(self, input_type='input', optimise=False, hessian=False, density=False, solvent=False,
                       restart=False, execute=True):
        """
        Generates the relevant job file for Gaussian, then executes this job file.
        :param input_type: The set of coordinates in the molecule that should be used in the job
        :param optimise: Optimise the geometry of the molecule
        :param hessian: Calculate the hessian matrix
        :param density: Calculate the electron density
        :param solvent: Use a solvent when calculating the electron density
        :param restart: Restart from a check point file
        :param execute: Run the calculation after writing the input file
        :return: The exit status of the job if ran, True for normal false for not ran or error
        """

        molecule = self.molecule.coords[input_type]

        with open(f'gj_{self.molecule.name}.com', 'w+') as input_file:

            input_file.write(f'%Mem={self.molecule.memory}GB\n%NProcShared={self.molecule.threads}\n%Chk=lig\n')

            commands = f'# {self.molecule.theory}/{self.molecule.basis} SCF=XQC '

            # Adds the commands in groups. They MUST be in the right order because Gaussian.
            if optimise:
                convergence = self.convergence_dict.get(self.molecule.convergence, "")
                if convergence != "":
                    convergence = f', {convergence}'
                # Set the convergence and the iteration cap for the optimisation
                commands += f'opt(MaxCycles={self.molecule.iterations} {convergence}) '

            if hessian:
                commands += 'freq '

            if solvent:
                commands += 'SCRF=(IPCM,Read) '

            if density:
                commands += 'density=current OUTPUT=WFX '

            if restart:
                commands += 'geom=check'

            commands += f'\n\n{self.molecule.name}\n\n{self.molecule.charge} {self.molecule.multiplicity}\n'

            input_file.write(commands)

            if not restart:
                # Add the atomic coordinates if we are not restarting from the chk file
                for i, atom in enumerate(molecule):
                    input_file.write(f'{self.molecule.atoms[i].element} {float(atom[0]): .10f} {float(atom[1]): .10f} '
                                     f'{float(atom[2]): .10f}\n')

            if solvent:
                # Adds the epsilon and cavity params
                input_file.write('\n4.0 0.0004')

            if density:
                # Specify the creation of the wavefunction file
                input_file.write(f'\n{self.molecule.name}.wfx')

            # Blank lines because Gaussian.
            input_file.write('\n\n')

        if execute:
            with open('log.txt', 'w+') as log:
                sp.run(f'g09 < gj_{self.molecule.name}.com > gj_{self.molecule.name}.log',
                       shell=True, stdout=log, stderr=log)

            # Now check the exit status of the job
            return self.check_for_errors()

        else:
            return {'success': False, 'error': 'Not run'}

    def check_for_errors(self):
        """
        Read the output file and check for normal termination and any errors.
        :return: A dictionary of the success status and any problems
        """

        with open(f'gj_{self.molecule.name}.log', 'r') as log:
            for line in log:
                if 'Normal termination of Gaussian' in line:
                    return {'success': True}

                elif 'Problem with the distance matrix.' in line:
                    return {'success': False,
                            'error': 'Distance matrix'}

                elif 'Error termination in NtrErr' in line:
                    return {'success': False,
                            'error': 'FileIO'}
                else:
                    return {'success': False,
                            'error': 'Unknown'}

    def hessian(self):
        """Extract the Hessian matrix from the Gaussian fchk file."""

        # Make the fchk file first
        with open('formchck.log', 'w+') as formlog:
            sp.run('formchk lig.chk lig.fchk', shell=True, stdout=formlog, stderr=formlog)

        with open('lig.fchk', 'r') as fchk:

            lines = fchk.readlines()
            hessian_list = []

            for count, line in enumerate(lines):
                if line.startswith('Cartesian Force Constants'):
                    start_pos = count + 1
                if line.startswith('Dipole Moment'):
                    end_pos = count

            if not start_pos and end_pos:
                raise EOFError('Cannot locate Hessian matrix in lig.fchk file.')

            for line in lines[start_pos: end_pos]:
                # Extend the list with the converted floats from the file, splitting on spaces and removing '\n' tags.
                hessian_list.extend([float(num) * 627.509391 / (0.529 ** 2) for num in line.strip('\n').split()])

        hess_size = 3 * len(self.molecule.atoms)

        hessian = np.zeros((hess_size, hess_size))

        # Rewrite Hessian to full, symmetric 3N * 3N matrix rather than list with just the non-repeated values.
        m = 0
        for i in range(hess_size):
            for j in range(i + 1):
                hessian[i, j] = hessian_list[m]
                hessian[j, i] = hessian_list[m]
                m += 1

        check_symmetry(hessian)

        return hessian

    def optimised_structure(self):
        """Extract the optimised structure from the Gaussian log file."""

        with open(f'gj_{self.molecule.name}.log', 'r') as log_file:

            lines = log_file.readlines()

        output = ''
        start, end, energy = None, None, None
        # Look for the output stream
        # TODO Escape sequence warnings. Just use r'' for search strings with \ in them
        for pos, line in enumerate(lines):
            if f'R{self.molecule.theory}\{self.molecule.basis}' in line:
                start = pos

            elif '@' in line:
                end = pos

            elif 'SCF Done' in line:
                energy = float(line.split()[4])

        if any(i is None for i in [start, end, energy]):
            raise EOFError('Cannot locate optimised structure in file.')

        # now add the lines to the output stream
        for line in range(start, end):
            output += lines[line].strip()

        # Split the string by the double slash to now find the molecule input
        molecule = []
        output = output.split('\\\\')
        for string in output:
            if string.startswith(f'{self.molecule.charge},{self.molecule.multiplicity}\\'):
                # Remove the charge and multiplicity from the string
                molecule = string.split('\\')[1:]

            # Store the coords back into the molecule array
            opt_struct = []
            for atom in molecule:
                atom = atom.split(",")
                opt_struct.append([float(atom[1]), float(atom[2]), float(atom[3])])

        return np.array(opt_struct), energy

    def all_modes(self):
        """Extract the frequencies from the Gaussian log file."""

        with open(f'gj_{self.molecule.name}.log', 'r') as gj_log_file:

            lines = gj_log_file.readlines()
            freqs = []

            # Stores indices of rows which will be used
            freq_positions = []
            for count, line in enumerate(lines):
                if line.startswith(' Frequencies'):
                    freq_positions.append(count)

            for pos in freq_positions:
                freqs.extend(float(num) for num in lines[pos].split()[2:])

        return np.array(freqs)