#!/usr/bin/env python3

# TODO Expand the functional_dict for PSI4 and Gaussian classes to "most" functionals.
# TODO Add better error handling for missing info. Maybe add path checking for Chargemol?
# TODO Rewrite file parsers to use takewhile / dropwhile / flagging rather than reading the whole files to memory.

from QUBEKit.helpers import get_overage, check_symmetry, append_to_log
from QUBEKit.decorators import for_all_methods, timer_logger

import subprocess as sp
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull

import qcengine as qcng
import qcelemental as qcel

from rdkit.Chem import AllChem, MolFromPDBFile, Descriptors, MolToSmiles, MolToSmarts, MolToMolFile, MolFromMol2File, MolFromMolFile, rdPartialCharges
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

import xml.etree.ElementTree as ET


class Engines:
    """
    Engines superclass containing core information that all other engines (PSI4, Gaussian etc) will have.
    Provides atoms' coordinates with name tags for each atom and entire molecule.
    Also gives all configs from the appropriate config file.
    """

    def __init__(self, molecule):

        self.molecule = molecule

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'


@for_all_methods(timer_logger)
class PSI4(Engines):
    """
    Writes and executes input files for psi4.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule):

        super().__init__(molecule)

        self.functional_dict = {'pbepbe': 'PBE', 'wb97xd': 'wB97X-D'}
        # Search for functional in dict, if it's not there, just leave the theory as it is.
        self.molecule.theory = self.functional_dict.get(self.molecule.theory, self.molecule.theory)

    # TODO add restart from log method
    def generate_input(self, input_type='input', optimise=False, hessian=False, density=False, energy=False,
                       fchk=False, restart=False, execute=True):
        """
        Converts to psi4 input format to be run in psi4 without using geometric.
        :param input_type: The coordinate set of the molecule to be used
        :param optimise: Optimise the molecule to the desired convergence critera with in the iteration limit
        :param hessian: Calculate the hessian matrix
        :param density: Calculate the electron density
        :param energy: Calculate the single point energy of the molecule
        :param fchk: Write out a gaussian style Fchk file
        :param restart: Restart the calculation from a log point
        :param execute: Run the desired Psi4 job
        :return: The completion status of the job True if successful False if not run or failed
        """

        molecule = self.molecule.coords[input_type]

        setters = ''
        tasks = ''

        # input.dat is the PSI4 input file.
        with open('input.dat', 'w+') as input_file:
            # opening tag is always writen
            input_file.write(f'memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n'
                             f'{self.molecule.charge} {self.molecule.multiplicity} \n')
            # molecule is always printed
            for i, atom in enumerate(molecule):
                input_file.write(f' {self.molecule.atoms[i].element}    {float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f} \n')
            input_file.write(f" units angstrom\n no_reorient\n}}\n\nset {{\n basis {self.molecule.basis}\n")

            if energy:
                append_to_log('Writing psi4 energy calculation input')
                tasks += f"\nenergy  = energy('{self.molecule.theory}')"

            if optimise:
                append_to_log('Writing PSI4 optimisation input', 'minor')
                setters += f' g_convergence {self.molecule.convergence}\n GEOM_MAXITER {self.molecule.iterations}\n'
                tasks += f"\noptimize('{self.molecule.theory.lower()}')"

            if hessian:
                append_to_log('Writing PSI4 Hessian matrix calculation input', 'minor')
                setters += ' hessian_write on\n'

                tasks += f"\nenergy, wfn = frequency('{self.molecule.theory.lower()}', return_wfn=True)"

                tasks += '\nwfn.hessian().print_out()\n\n'

            if density:
                append_to_log('Writing PSI4 density calculation input', 'minor')
                setters += " cubeprop_tasks ['density']\n"

                overage = get_overage(self.molecule.name)
                setters += ' CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n'.format(overage)
                setters += ' CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n'
                tasks += f"grad, wfn = gradient('{self.molecule.theory.lower()}', return_wfn=True)\ncubeprop(wfn)"

            if fchk:
                append_to_log('Writing PSI4 input file to generate fchk file')
                tasks += f"\ngrad, wfn = gradient('{self.molecule.theory.lower()}', return_wfn=True)"
                tasks += '\nfchk_writer = psi4.core.FCHKWriter(wfn)'
                tasks += f'\nfchk_writer.write("{self.molecule.name}_psi4.fchk")\n'

            # TODO If overage cannot be made to work, delete and just use Gaussian.
            # if self.molecule.solvent:
            #     setters += ' pcm true\n pcm_scf_type total\n'
            #     tasks += '\n\npcm = {'
            #     tasks += '\n units = Angstrom\n Medium {\n  SolverType = IEFPCM\n  Solvent = Chloroform\n }'
            #     tasks += '\n Cavity {\n  RadiiSet = UFF\n  Type = GePol\n  Scaling = False\n  Area = 0.3\n  Mode = Implicit'
            #     tasks += '\n }\n}'

            setters += '}\n'

            if not execute:
                setters += f'set_num_threads({self.molecule.threads})\n'

            input_file.write(setters)
            input_file.write(tasks)

        if execute:
            with open('log.txt', 'w+') as log:
                sp.run(f'psi4 input.dat -n {self.molecule.threads}', shell=True, stdout=log, stderr=log)

            # After running, check for normal termination
            return True if '*** Psi4 exiting successfully.' in open('output.dat', 'r').read() else False
        else:
            return False

    def hessian(self):
        """
        Parses the Hessian from the output.dat file (from psi4) into a numpy array.
        Molecule is a numpy array of size N x N.
        """

        hess_size = 3 * len(self.molecule.atoms)

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:

            lines = file.readlines()

            for count, line in enumerate(lines):
                if '## Hessian' in line or '## New Matrix (Symmetry' in line:
                    # Set the start of the hessian to the row of the first value.
                    hess_start = count + 5
                    break
            else:
                raise EOFError('Cannot locate Hessian matrix in output.dat file.')

            # Check if the hessian continues over onto more lines (i.e. if hess_size is not divisible by 5)
            extra = 0 if hess_size % 5 == 0 else 1

            # hess_length: # of cols * length of each col
            #            + # of cols - 1 * #blank lines per row of hess_vals
            #            + # blank lines per row of hess_vals if the hess_size continues over onto more lines.
            hess_length = (hess_size // 5) * hess_size + (hess_size // 5 - 1) * 3 + extra * (3 + hess_size)

            hess_end = hess_start + hess_length

            hess_vals = []

            for file_line in lines[hess_start:hess_end]:
                # Compile lists of the 5 Hessian floats for each row.
                # Number of floats in last row may be less than 5.
                # Only the actual floats are added, not the separating numbers.
                row_vals = [float(val) for val in file_line.split() if len(val) > 5]
                hess_vals.append(row_vals)

            # Remove blank list entries
            hess_vals = [elem for elem in hess_vals if elem]

            reshaped = []

            # Convert from list of (lists, length 5) to 2d array of size hess_size x hess_size
            for old_row in range(hess_size):
                new_row = []
                for col_block in range(hess_size // 5 + extra):
                    new_row += hess_vals[old_row + col_block * hess_size]

                reshaped.append(new_row)

            hess_matrix = np.array(reshaped)

            # Cache the unit conversion.
            conversion = 627.509391 / (0.529 ** 2)
            # Element-wise multiplication
            hess_matrix *= conversion

            check_symmetry(hess_matrix)

            return hess_matrix

    def optimised_structure(self):
        """
        Parses the final optimised structure from the output.dat file (from psi4) to a numpy array.
        Also returns the energy of the optimized structure.
        """

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if '==> Geometry' in line:
                    geo_pos_list.append(count)

                elif '**** Optimization is complete!' in line:
                    opt_pos = count
                    opt_steps = int(line.split()[5])

            if not (opt_pos and opt_steps):
                raise EOFError('According to the output.dat file, optimisation has not completed.')

            # now get the final opt_energy
            opt_energy = float(lines[opt_pos + opt_steps + 7].split()[1])

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            opt_struct = []

            for row in range(len(self.molecule.atoms)):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = []
                for indx in range(3):
                    struct_row.append(float(lines[start_of_vals + row].split()[indx + 1]))

                opt_struct.append(struct_row)

        return np.array(opt_struct), opt_energy

    @staticmethod
    def get_energy():
        """Get the energy of a single point calculation."""

        # open the psi4 log file
        with open('output.dat', 'r') as log:
            lines = log.readlines()

        # find the total converged energy
        for line in lines:
            if 'Total Energy =' in line:
                energy = float(line.split()[3])
                break
        else:
            raise EOFError('Cannot find energy in output.dat file.')

        return energy

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            for count, line in enumerate(lines):
                if "post-proj  all modes" in line:
                    start_of_vals = count
                    break
            else:
                raise EOFError('Cannot locate modes in output.dat file.')

            # Barring the first (and sometimes last) line, dat file has 6 values per row.
            end_of_vals = start_of_vals + (3 * len(self.molecule.atoms)) // 6

            structures = lines[start_of_vals][24:].replace("'", "").split()
            structures = structures[6:]

            for row in range(1, end_of_vals - start_of_vals):
                # Remove double strings and weird formatting.
                structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

            all_modes = [float(val) for val in structures]

            return np.array(all_modes)

    def geo_gradient(self, input_type='input', threads=False, execute=True):
        """
        Write the psi4 style input file to get the gradient for geometric
        and run geometric optimisation.
        """

        molecule = self.molecule.coords[input_type]

        with open(f'{self.molecule.name}.psi4in', 'w+') as file:

            file.write(f'memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n {self.molecule.charge} {self.molecule.multiplicity} \n')
            for i, atom in enumerate(molecule):
                file.write(f'  {self.molecule.atoms[i].element:2}    {float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f}\n')

            file.write(f' units angstrom\n no_reorient\n}}\nset basis {self.molecule.basis}\n')

            if threads:
                file.write(f'set_num_threads({self.molecule.threads})')
            file.write(f"\n\ngradient('{self.molecule.theory}')\n")

        if execute:
            with open('log.txt', 'w+') as log:
                sp.run(f'geometric-optimize --psi4 {self.molecule.name}.psi4in {self.molecule.constraints_file} '
                       f'--nt {self.molecule.threads}', shell=True, stdout=log, stderr=log)


@for_all_methods(timer_logger)
class Chargemol(Engines):

    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_input(self, execute=True):
        """Given a DDEC version (from the defaults), this function writes the job file for chargemol and executes it."""

        if (self.molecule.ddec_version != 6) and (self.molecule.ddec_version != 3):
            append_to_log(message='Invalid or unsupported DDEC version given, running with default version 6.',
                          msg_type='warning')

        # Write the charges job file.
        with open('job_control.txt', 'w+') as charge_file:

            charge_file.write(f'<input filename>\n{self.molecule.name}.wfx\n</input filename>')

            charge_file.write('\n\n<net charge>\n0.0\n</net charge>')

            charge_file.write('\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.')
            charge_file.write('\n</periodicity along A, B and C vectors>')

            charge_file.write(f'\n\n<atomic densities directory complete path>\n{self.molecule.chargemol}'
                              f'/atomic_densities/')
            charge_file.write('\n</atomic densities directory complete path>')

            charge_file.write(f'\n\n<charge type>\nDDEC{self.molecule.ddec_version}\n</charge type>')

            charge_file.write('\n\n<compute BOs>\n.true.\n</compute BOs>')

        if execute:
            with open('log.txt', 'w+') as log:
                # TODO path.join()?
                control_path = 'chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/' \
                               'Chargemol_09_26_2017_linux_serial job_control.txt'
                sp.run(f'{self.molecule.chargemol}/{control_path}', shell=True, stdout=log, stderr=log)


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
        :return: A dictionary with the success or error and the type
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
                    input_file.write(f'{self.molecule.atoms[i].element}  {float(atom[0]): .10f} {float(atom[1]): .10f} '
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
            result = self.check_for_errors()

            return result

        else:

            result = {'success': False,
                      'error': 'Not run'}

            return result

    def check_for_errors(self):
        """
        Read the output file and check for normal termination and any errors.
        :return: A dictionary of the success status and any problems
        """

        log = open(f'gj_{self.molecule.name}.log', 'r').read()
        if 'Normal termination of Gaussian' in log:
            return {'success': True}

        elif 'Problem with the distance matrix.' in log:
            return {'success': False,
                    'error': 'Distance matrix'}

        elif 'Error termination in NtrErr' in log:
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
        """
        Extract the optimised structure and energy from a fchk file
        :return molecule: The optimised array with the structure
        :return energy:  The SCF energy of the optimised structure
        """
        # Make the fchk file first
        with open('formchck.log', 'w+') as formlog:
            sp.run('formchk lig.chk lig.fchk', shell=True, stdout=formlog, stderr=formlog)

        with open('lig.fchk', 'r') as fchk:

            lines = fchk.readlines()

        start, end, energy = None, None, None

        for i, line in enumerate(lines):
            if 'Current cartesian coordinates' in line:
                start = i + 1
            elif 'Int Atom Types' in line:
                end = i - 1
            elif 'Total Energy' in line:
                energy = float(line.split()[3])

        if any(x is None for x in [start, end, energy]):
            raise EOFError('Cannot locate optimised structure in file.')

        molecule = []
        # Now get the coords from the file
        for line in lines[start: end]:
            molecule.extend([float(coord) for coord in line.split()])

        molecule = np.round(np.array(molecule).reshape((len(self.molecule.atoms), 3)) * 0.529, decimals=10)

        return molecule, energy

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

# TODO do we need this class anymore it only makes an xyz file? Maybe move calculate_hull to helpers
@for_all_methods(timer_logger)
class ONETEP(Engines):

    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_input(self, input_type='input', density=False):
        """ONETEP takes a xyz input file."""

        if density:
            self.molecule.write_xyz(input_type=input_type)

        # should we make a onetep run file? this is quite specific?
        print('Run this file in ONETEP.')

    def calculate_hull(self):
        """
        Generate the smallest convex hull which encloses the molecule.
        Then make a 3d plot of the points and hull.
        """

        coords = np.array([atom[1:] for atom in self.molecule.coords['input']])

        hull = ConvexHull(coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        ax.plot(coords.T[0], coords.T[1], coords.T[2], 'ko')

        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(coords[simplex, 0], coords[simplex, 1], coords[simplex, 2], color='lightseagreen')

        plt.show()


@for_all_methods(timer_logger)
class QCEngine(Engines):

    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_qschema(self, input_type='input'):
        """
        Using the molecule object, generate a QCEngine schema. This can then
        be fed into the various QCEngine procedures.
        :param input_type: The part of the molecule object that should be used when making the schema
        :return: The qcelemental qschema
        """

        mol_data = f'{self.molecule.charge} {self.molecule.multiplicity}\n'

        for i, coord in enumerate(self.molecule.coords[input_type]):
            mol_data += f'{self.molecule.atoms[i].element} '
            for item in coord:
                mol_data += f'{item} '
            mol_data += '\n'

        return qcel.models.Molecule.from_data(mol_data)

    def call_qcengine(self, engine, driver, input_type):
        """
        Using the created schema, run a particular engine, specifying the driver (job type).
        e.g. engine: geo, driver: energies.
        :param engine: The engine to be used psi4 geometric
        :param driver: The calculation type to be done e.g. energy, gradient, hessian, properties
        :param input_type: The part of the molecule object that should be used when making the schema
        :return: The required driver information
        """

        mol = self.generate_qschema(input_type=input_type)

        # Call psi4 for energy, gradient, hessian or property calculations
        if engine == 'psi4':
            psi4_task = qcel.models.ResultInput(
                molecule=mol,
                driver=driver,
                model={'method': self.molecule.theory, 'basis': self.molecule.basis},
                keywords={'scf_type': 'df'},
            )

            ret = qcng.compute(psi4_task, 'psi4', local_options={'memory': self.molecule.memory,
                                                                 'ncores': self.molecule.threads})

            if driver == 'hessian':
                hess_size = 3 * len(self.molecule.atoms)
                hessian = np.reshape(ret.return_result, (hess_size, hess_size)) * 627.509391 / (0.529 ** 2)
                check_symmetry(hessian)

                return hessian

            else:
                return ret.return_result

        # Call geometric with psi4 to optimise a molecule
        elif engine == 'geometric':
            geo_task = {
                'schema_name': 'qcschema_optimization_input',
                'schema_version': 1,
                'keywords': {
                    'coordsys': 'tric',
                    'maxiter': self.molecule.iterations,
                    'program': 'psi4',
                    'convergence_set': self.molecule.convergence,
                },
                'input_specification': {
                    'schema_name': 'qcschema_input',
                    'schema_version': 1,
                    'driver': 'gradient',
                    'model': {'method': self.molecule.theory, 'basis': self.molecule.basis},
                    'keywords': {},
                },
                'initial_molecule': mol,
            }
            ret = qcng.compute_procedure(
                geo_task, 'geometric', return_dict=True, local_options={'memory': self.molecule.memory,
                                                                        'ncores': self.molecule.threads})
            return ret

        else:
            raise KeyError('Invalid engine type provided. Please use "geo" or "psi4".')


@for_all_methods(timer_logger)
class RDKit:
    """Class for controlling useful RDKit functions; try to keep class static."""

    @staticmethod
    def read_file(filename):

        # This handles splitting the paths
        filename = Path(filename)

        # Try and read the file
        if filename.suffix == '.pdb':
            mol = MolFromPDBFile(filename.name, removeHs=False)
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except RuntimeError:
                print('RDKit could not assign the partial charges')
        elif filename.suffix == '.mol2':
            mol = MolFromMol2File(filename.name, removeHs=False)
        elif filename.suffix == '.mol':
            mol = MolFromMolFile(filename.name, removeHs=False)
        else:
            mol = None

        return mol

    @staticmethod
    def smiles_to_pdb(smiles_string, name=None):
        """
        Converts smiles strings to RDKit molobject.
        :param smiles_string: The hydrogen free smiles string
        :param name: The name of the molecule this will be used when writing the pdb file
        :return: The RDKit molecule
        """
        # Originally written by venkatakrishnan; rewritten and extended by Chris Ringrose

        if 'H' in smiles_string:
            raise SyntaxError('Smiles string contains hydrogen atoms; try again.')

        m = AllChem.MolFromSmiles(smiles_string)
        if name is None:
            name = input('Please enter a name for the molecule:\n>')
        m.SetProp('_Name', name)
        mol_hydrogens = AllChem.AddHs(m)
        AllChem.EmbedMolecule(mol_hydrogens, AllChem.ETKDG())
        AllChem.SanitizeMol(mol_hydrogens)

        print(AllChem.MolToMolBlock(mol_hydrogens), file=open(f'{name}.mol', 'w+'))
        AllChem.MolToPDBFile(mol_hydrogens, f'{name}.pdb')

        return f'{name}.pdb'

    @staticmethod
    def mm_optimise(filename, ff='MMF'):
        """
        Perform rough preliminary optimisation to speed up later optimisations.
        :param filename: The name of the input file
        :param ff: The Force field to be used either MMF or UFF
        :return: The name of the optimised pdb file that is made
        """

        # Get the rdkit molecule
        mol = RDKit.read_file(filename)

        force_fields = {'MMF': MMFFOptimizeMolecule, 'UFF': UFFOptimizeMolecule}

        force_fields[ff](mol)

        AllChem.MolToPDBFile(mol, f'{filename.stem}_rdkit_optimised.pdb')

        return f'{filename.stem}_rdkit_optimised.pdb'

    @staticmethod
    def rdkit_descriptors(filename):
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary.
        :param filename: The molecule input file
        :return: Descriptors dictionary
        """

        mol = RDKit.read_file(filename)
        # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        descriptors = {'Heavy atoms': Descriptors.HeavyAtomCount(mol),
                       'H-bond donors': Descriptors.NumHDonors(mol),
                       'H-bond acceptors': Descriptors.NumHAcceptors(mol),
                       'Molecular weight': Descriptors.MolWt(mol),
                       'LogP': Descriptors.MolLogP(mol)}

        return descriptors

    @staticmethod
    def get_smiles(filename):
        """
        Use RDKit to load in the pdb file of the molecule and get the smiles code.
        :param filename: The molecule input file
        :return: The smiles string
        """

        mol = RDKit.read_file(filename)

        return MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)

    @staticmethod
    def get_smarts(filename):
        """
        Use RDKit to get the smarts string of the molecule.
        :param filename: The molecule input file
        :return: The smarts string
        """

        mol = RDKit.read_file(filename)

        return MolToSmarts(mol)

    @staticmethod
    def get_mol(filename):
        """
        Use RDKit to generate a mol file.
        :param filename: The molecule input file
        :return: The name of the mol file made
        """

        mol = RDKit.read_file(filename)

        mol_name = f'{filename.steam}.mol'
        MolToMolFile(mol, mol_name)

        return mol_name

    @staticmethod
    def generate_conformers(filename, conformer_no=10):
        """
        Generate a set of x conformers of the molecule
        :param conformer_no: The amount of conformers made for the molecule
        :param filename: The name of the input file
        :return: A list of conformer position arrays
        """

        mol = RDKit.read_file(filename)

        AllChem.EmbedMultipleConfs(mol, numConfs=conformer_no)
        positions = mol.GetConformers()
        coords = [conformer.GetPositions() for conformer in positions]

        return coords


@for_all_methods(timer_logger)
class OpenMM:
    """This class acts as a wrapper around OpenMM so we can many basic functions using the class"""

    def __init__(self, molecule):
        self.molecule = molecule
        self.system = None
        self.simulation = None
        self.combination = None
        self.pdb = molecule.name + '.pdb'
        self.xml = molecule.name + '.xml'
        self.openmm_system()

    def openmm_system(self):
        """Initialise the OpenMM system we will use to evaluate the energies."""

        # Load the initial coords into the system and initialise
        pdb = app.PDBFile(self.pdb)
        forcefield = app.ForceField(self.xml)
        modeller = app.Modeller(pdb.topology, pdb.positions)  # set the initial positions from the pdb
        self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        # Check what combination rule we should be using from the xml
        xmlstr = open(self.xml).read()
        # check if we have opls combination rules if the xml is present
        try:
            self.combination = ET.fromstring(xmlstr).find('NonbondedForce').attrib['combination']
            append_to_log('OPLS combination rules found in xml file', msg_type='minor')
        except AttributeError:
            pass
        except KeyError:
            pass

        if self.combination == 'opls':
            self.opls_lj()

        temperature = 298.15 * unit.kelvin
        integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)

        self.simulation = app.Simulation(modeller.topology, self.system, integrator)
        self.simulation.context.setPositions(modeller.positions)

    def get_energy(self, position, forces=False):
        """
        Return the MM calculated energy of the structure
        :param position: The OpenMM formatted atomic positions
        :param forces: If we should also get the forces
        :return:
        """

        # update the positions of the system
        self.simulation.context.setPositions(position)

        # Get the energy from the new state
        state = self.simulation.context.getState(getEnergy=True, getForces=forces)

        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        if forces:
            gradient = state.getForces(asNumpy=True)

            return energy, gradient

        return energy

    def opls_lj(self):
        """
        This function changes the standard OpenMM combination rules to use OPLS, execp and normal pairs are only
        required if their are virtual sites in the molecule.
        """

        # Get the system information from the openmm system
        forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in
                  range(self.system.getNumForces())}
        # Use the nondonded_force to get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = mm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
        lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        self.system.addForce(lorentz)

        l_j_set = {}
        # For each particle, calculate the combination list again
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            l_j_set[index] = (sigma, epsilon, charge)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(index, charge, 0, 0)

        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                charge = 0.5 * (l_j_set[p1][2] * l_j_set[p2][2])
                sig14 = np.sqrt(l_j_set[p1][0] * l_j_set[p2][0])
                nonbonded_force.setExceptionParameters(i, p1, p2, charge, sig14, eps)
            # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
            # Old method which needs updating
            # if excep_pairs:
            #     for x in range(len(excep_pairs)):  # scale 14 interactions
            #         if p1 == excep_pairs[x, 0] and p2 == excep_pairs[x, 1] or p2 == excep_pairs[x, 0] and p1 == \
            #                 excep_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2 * 0.5
            #             sig14 = sqrt(sigma1 * sigma2) * 0.5
            #             eps = sqrt(epsilon1 * epsilon2) * 0.5
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            #
            # if normal_pairs:
            #     for x in range(len(normal_pairs)):
            #         if p1 == normal_pairs[x, 0] and p2 == normal_pairs[x, 1] or p2 == normal_pairs[x, 0] and p1 == \
            #                 normal_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2
            #             sig14 = sqrt(sigma1 * sigma2)
            #             eps = sqrt(epsilon1 * epsilon2)
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

    def format_coords(self, coordinates):
        """
        Take the coordinates as a list and format to the OpenMM style of a list of tuples.
        :param coordinates: The flattened list of coordinates.
        :return: The OpenMM list of tuples.
        """

        coords = []
        for i in range(0, len(coordinates), 3):
            coords.append(tuple(coordinates[i:i+3]))

        return coords

    def calculate_hessian(self, finite_step):
        """
        Using finite displacement calculate the hessian matrix of the molecule using symmetric difference quotient (SQD) rule.
        :param finite_step: The finite step size used in the calculation in nm
        :return: A numpy array of the mass weighted hessian of size 3N*3N
        """

        # Create the OpenMM coords list from the qm coordinates and convert to nm
        input_coords = self.molecule.coords['qm'].flatten() / 10

        # We get each hessian element from = [E(dx + dy) + E(-dx - dy) - E(dx - dy) - E(-dx + dy)] / 4 dx dy
        hessian = np.zeros((3 * len(self.molecule.atoms), 3 * len(self.molecule.atoms)))

        for i in range(3 * len(self.molecule.atoms)):
            for j in range(i, 3 * len(self.molecule.atoms)):
                # Mutate the atomic coords
                # Do less energy evaluations on the diagonal of the matrix
                if i == j:
                    coords = deepcopy(input_coords)
                    coords[i] += 2 * finite_step
                    e1 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= 2 * finite_step
                    e2 = self.get_energy(self.format_coords(coords))
                    hessian[i, j] = (e1 + e2) / (4 * finite_step**2 * self.molecule.atoms[i // 3].mass)
                else:
                    coords = deepcopy(input_coords)
                    coords[i] += finite_step
                    coords[j] += finite_step
                    e1 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= finite_step
                    coords[j] -= finite_step
                    e2 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] += finite_step
                    coords[j] -= finite_step
                    e3 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= finite_step
                    coords[j] += finite_step
                    e4 = self.get_energy(self.format_coords(coords))
                    hessian[i, j] = (e1 + e2 - e3 - e4) / (4 * finite_step**2 * self.molecule.atoms[i // 3].mass)

        # Now make the matrix symmetric
        sym_hessian = hessian + hessian.T - np.diag(hessian.diagonal())
        return sym_hessian

    def normal_modes(self, finite_step):
        """
        Calculate the normal modes of the molecule from the hessian matrix
        :param finite_step: The finite step size used in the calculation of the matrix
        :return: A numpy array of the normal modes of the molecule
        """

        # Get the mass weighted hessian matrix in amu
        hessian = self.calculate_hessian(finite_step)

        # Now get the eigenvalues and vectors
        e_vals, e_vectors = np.linalg.eig(hessian)
        print(e_vals)
        print(e_vectors)


