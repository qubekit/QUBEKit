#!/usr/bin/env python


from QUBEKit.helpers import get_overage, check_symmetry
from QUBEKit.decorators import for_all_methods, timer_logger

from subprocess import call as sub_call
from numpy import array, zeros


class Engines:
    """Engines superclass containing core information that all other engines (PSI4, Gaussian etc) will have.
    Provides atoms' coordinates with name tags for each atom and entire molecule.
    Also gives all configs from the appropriate config file.
    """

    def __init__(self, molecule, config_dict):

        self.engine_mol = molecule
        self.charge = config_dict[0]['charge']
        self.multiplicity = config_dict[0]['multiplicity']
        # Load the configs using the config_file name from the csv.
        self.qm, self.fitting, self.descriptions = config_dict[1], config_dict[2], config_dict[3]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'


@for_all_methods(timer_logger)
class PSI4(Engines):
    """Psi4 class (child of Engines).
    Used to extract optimised structures, Hessians, frequencies, etc.
    Writes and executes input files for psi4.
    """

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

        self.functional_dict = {'PBEPBE': 'PBE'}
        if self.qm['theory'] in list(self.functional_dict.keys()):
            self.qm['theory'] = self.functional_dict[self.qm['theory']]

    def hessian(self):
        """Parses the Hessian from the *_output.dat file (from psi4) into a numpy array.
        Molecule is a numpy array of size N x N.
        """

        hess_size = 3 * len(self.engine_mol.molecule)

        # *_output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:

            lines = file.readlines()

            for count, line in enumerate(lines):

                if '## Hessian' in line:
                    # Set the start of the hessian to the row of the first value.
                    hess_start = count + 5

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

                    check_symmetry(reshaped)

                    # Units conversion.
                    hess_matrix = array(reshaped) * 627.509391 / (0.529 ** 2)

                    print(f'Extracted Hessian for {self.engine_mol.name} from psi4 output')
                    self.engine_mol.hessian = hess_matrix

                    return self.engine_mol

    def optimised_structure(self):
        """Parses the final optimised structure from the *_output.dat file (from psi4) to a numpy array."""

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # *_output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if "==> Geometry" in line:
                    geo_pos_list.append(count)

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            opt_struct = []

            for row in range(len(self.engine_mol.molecule)):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = [lines[start_of_vals + row].split()[0]]
                for indx in range(1, 4):
                    struct_row.append(float(lines[start_of_vals + row].split()[indx]))

                opt_struct.append(struct_row)

        print(f'Extracted optimised structure for {self.engine_mol.name} from psi4 output')
        self.engine_mol.QMoptimized = opt_struct

        return self.engine_mol

    def energy(self):
        pass

    def generate_input(self, QM=False, MM=False, optimize=False, hessian=False, density=False, threads=False):
        """Converts to psi4 input format to be run in psi4 without using geometric"""

        if QM:
            molecule = self.engine_mol.QMoptimized
        elif MM:
            molecule = self.engine_mol.MMoptimized
        else:
            molecule = self.engine_mol.molecule

        # input.dat is the psi4 input file.
        setters = ''
        tasks = ''

        # opening tag is always writen
        with open('input.dat', 'w+') as input_file:
            input_file.write('memory {} GB\n\nmolecule {} {{\n{} {} \n'.format(self.qm['threads'], self.engine_mol.name,
                                                                               self.charge, self.multiplicity))
            # molecule is always printed
            for atom in molecule:
                input_file.write(' {}    {: .10f}  {: .10f}  {: .10f} \n'.format(atom[0], float(atom[1]), float(atom[2]), float(atom[3])))
            input_file.write(' units angstrom\n no_reorient\n}}\n\nset {{\n basis {}\n'.format(self.qm['basis']))

            if optimize:
                print('Writing psi4 optimisation input')
                setters += ' g_convergence {}\n GEOM_MAXITER {}\n'.format(self.qm['convergence'], self.qm['iterations'])
                tasks += "\noptimize('{}')".format(self.qm['theory'].lower())

            if hessian:
                print('Writing psi4 hessian calculation input')
                setters += ' hessian_write on\n'

                tasks += "\nenergy, wfn = frequency('{}', return_wfn=True)".format(self.qm['theory'].lower())

                tasks += '\nwfn.hessian().print_out()\n\n'

            if density:
                print('Writing psi4 density calculation input')
                setters += " cubeprop_tasks ['density']\n"
                # TODO Handle overage correctly (should be dependent on the size of the molecule).
                # See helpers.get_overage for info.

                # print('Calculating overage for psi4 and chargemol.')
                overage = get_overage(self.engine_mol.name)
                setters += " CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n".format(overage)
                setters += " CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n"
                tasks += "grad, wfn = gradient('{}', return_wfn=True)\ncubeprop(wfn)".format(self.qm['theory'].lower())

            # TODO If overage cannot be made to work, delete and just use Gaussian.
            # if self.qm['solvent']:
            #     print('Setting pcm parameters.')
            #     setters += ' pcm true\n pcm_scf_type total\n'
            #     tasks += '\n\npcm = {'
            #     tasks += '\n units = Angstrom\n Medium {\n  SolverType = IEFPCM\n  Solvent = Chloroform\n }'
            #     tasks += '\n Cavity {\n  RadiiSet = UFF\n  Type = GePol\n  Scaling = False\n  Area = 0.3\n  Mode = Implicit'
            #     tasks += '\n }\n}'

            setters += '}\n'

            if threads:
                setters += f'set_num_threads({self.qm["threads"]})\n'

            input_file.write(setters)
            input_file.write(tasks)

        print('Runing quantum calculation using Psi4')
        sub_call(f'psi4 input.dat -n {self.qm["threads"]}', shell=True)

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # *_output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            for count, line in enumerate(lines):
                if "post-proj  all modes" in line:
                    start_of_vals = count

                    # Barring the first (and sometimes last) line, dat file has 6 values per row.
                    end_of_vals = count + (3 * len(self.engine_mol.molecule)) // 6

                    structures = lines[start_of_vals][24:].replace("'", "").split()
                    structures = structures[6:]

                    for row in range(1, end_of_vals - start_of_vals):
                        # Remove double strings and weird formatting.
                        structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

                    all_modes = [float(val) for val in structures]
                    self.engine_mol.modes = array(all_modes)

                    return self.engine_mol

    def geo_gradient(self, QM=False, MM=False, run=True, threads=False):
        """Write the psi4 style input file to get the gradient for geometric
        and run geometric optimisation.
        """

        if QM:
            molecule = self.engine_mol.QMoptimized
        elif MM:
            molecule = self.engine_mol.MMoptimized
        else:
            molecule = self.engine_mol.molecule

        with open(f'{self.engine_mol.name}.psi4in', 'w+') as file:

            file.write('molecule {} {{\n {} {} \n'.format(self.engine_mol.name, self.charge, self.multiplicity))
            for atom in molecule:
                file.write('  {}    {: .10f}  {: .10f}  {: .10f}\n'.format(atom[0], float(atom[1]), float(atom[2]), float(atom[3])))

            file.write("units angstrom\n no_reorient\n}}\nset basis {}\n".format(self.qm['basis']))

            if threads:
                file.write('set_num_threads({})'.format(self.qm['threads']))
            file.write("\n\ngradient('{}')\n".format(self.qm['theory']))

        if run:
            print('Optimizing molecule using Psi4 and geometric')
            with open('log.txt', 'w+') as log:
                sub_call(f'geometric-optimize --psi4 {self.engine_mol.name}.psi4in --nt {self.qm["threads"]}',
                         shell=True, stdout=log)
        else:
            print('Geometric psi4 optimise file written')


@for_all_methods(timer_logger)
class Chargemol(Engines):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)

    def generate_input(self):
        """Given a DDEC version (from the defaults), this function writes the job file for chargemol and
        executes it.
        """

        if (self.qm['ddec_version'] != 6) and (self.qm['ddec_version'] != 3):
            print('Invalid or unsupported DDEC version given, running with default version 6.')
            self.qm['ddec_version'] = 6

        # Write the charges job file.
        with open('job_control.txt', 'w+') as charge_file:

            # charge_file.write(f'<input filename>\n{self.engine_mol.name}.wfx\n</input filename>')
            charge_file.write(f'<input filename>\n{self.engine_mol.name}.wfx\n</input filename>')

            charge_file.write('\n\n<net charge>\n0.0\n</net charge>')

            charge_file.write('\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.')
            charge_file.write('\n</periodicity along A, B and C vectors>')

            charge_file.write(f'\n\n<atomic densities directory complete path>\n{self.descriptions["chargemol"]}/atomic_densities/')
            charge_file.write('\n</atomic densities directory complete path>')

            charge_file.write(f'\n\n<charge type>\nDDEC{self.qm["ddec_version"]}\n</charge type>')

            charge_file.write('\n\n<compute BOs>\n.true.\n</compute BOs>')

        # sub_call(f'psi4 input.dat -n {self.qm["threads"]}', shell=True)
        # sub_call('mv Dt.cube total_density.cube', shell=True)

        print(f'Partitioning charges with DDEC{self.qm["ddec_version"]}')
        sub_call(f'{self.descriptions["chargemol"]}/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_serial job_control.txt',
                 shell=True)

    def extract_charges(self):
        """Extract the charge data from the chargemol execution.
        Currently this is done by the LennardJones class.
        """

        pass


@for_all_methods(timer_logger)
class Gaussian(Engines):

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

        self.functional_dict = {'PBE': 'PBEPBE'}
        if self.qm['theory'] in list(self.functional_dict.keys()):
            self.qm['theory'] = self.functional_dict[self.qm['theory']]

    def generate_input(self, QM=False, MM=False, optimize=False, hessian=False, density=False, solvent=False):
        """Generates the relevant job file for Gaussian, then executes this job file."""

        if QM:
            molecule = self.engine_mol.QMoptimized
        elif MM:
            molecule = self.engine_mol.MMoptimized
        else:
            molecule = self.engine_mol.molecule

        with open(f'gj_{self.engine_mol.name}', 'w+') as input_file:

            input_file.write(f'%Mem={self.qm["memory"]}GB\n%NProcShared={self.qm["threads"]}\n%Chk=lig\n')

            commands = f'# {self.qm["theory"]}/{self.qm["basis"]} SCF=XQC '

            # Adds the commands in groups. They MUST be in the right order because Gaussian.

            if optimize:
                commands += 'opt '

            if hessian:
                commands += 'freq '

            if self.qm['solvent']:
                commands += 'SCRF=(IPCM,Read) '

            if density:
                commands += 'density=current OUTPUT=WFX '

            commands += f'\n\n{self.engine_mol.name}\n\n{self.charge} {self.multiplicity}\n'

            input_file.write(commands)

            # Add the atomic coordinates
            for atom in molecule:
                input_file.write('{} {: .3f} {: .3f} {: .3f}\n'.format(atom[0], float(atom[1]), float(atom[2]), float(atom[3])))

            if solvent:
                # Adds the epsilon and cavity params
                input_file.write('\n4.0 0.0004')

            if density:
                # Specify the creation of the wavefunction file
                input_file.write(f'\n{self.engine_mol.name}.wfx')

            # Blank lines because Gaussian.
            input_file.write('\n\n')

        print('Running Gaussian09 analysis')
        sub_call(f'g09 < gj_{self.engine_mol.name} > gj_{self.engine_mol.name}.log', shell=True)

    def optimised_structure(self):
        """Extract the optimised structure from the Gaussian log file."""

        with open(f'gj_{self.engine_mol.name}.log', 'r') as log_file:

            lines = log_file.readlines()

            opt_coords_pos = []
            for count, line in enumerate(lines):
                if 'Input orientation' in line:
                    opt_coords_pos.append(count + 5)
            start_pos = opt_coords_pos[-1]

            num_atoms = len(self.engine_mol.molecule)

            opt_struct = []

            for line in lines[start_pos: start_pos + num_atoms]:
                for atom_index in range(num_atoms):
                    # Takes atom name from molecule object and appends the *unpacked coordinates from the log file.
                    opt_struct.append([self.engine_mol.molecule[atom_index][0], *line.split()[-3:]])

        print(f'Extracted optimised structure for {self.engine_mol.name} from Gaussian log file.')
        return array(opt_struct)

    def hessian(self):
        """Extract the Hessian matrix from the Gaussian fchk file."""

        with open('lig.fchk', 'r') as fchk:

            lines = fchk.readlines()
            hessian_list = []

            for count, line in enumerate(lines):
                if line.startswith('Cartesian Force Constants'):
                    start_pos = count + 1
                if line.startswith('Dipole Moment'):
                    end_pos = count

            for line in lines[start_pos: end_pos]:
                # Extend the list with the converted floats from the file, splitting on spaces and removing '\n' tags.
                hessian_list.extend([float(num) * 0.529 for num in line.strip('\n').split()])

        hess_size = 3 * len(self.engine_mol.molecule)

        hessian = zeros((hess_size, hess_size))

        # Rewrite Hessian to full, symmetric 3N * 3N matrix rather than list with just the non-repeated values.
        m = 0
        for i in range(hess_size):
            for j in range(i + 1):
                hessian[i][j] = hessian_list[m]
                hessian[j][i] = hessian_list[m]
                m += 1

        check_symmetry(hessian)

        print(f'Extracted Hessian matrix for {self.engine_mol.name} from Gaussian fchk file.')
        return hessian

    def all_modes(self):
        """Extract the frequencies from the Gaussian log file."""

        with open(f'gj_{self.engine_mol.name}.log', 'r') as log_file:

            lines = log_file.readlines()
            freqs = []

            # Stores indices of rows which will be used
            freq_positions = []
            for count, line in enumerate(lines):
                if line.startswith(' Frequencies'):
                    freq_positions.append(count)

            for pos in freq_positions:
                freqs.extend(float(num) for num in lines[pos].split()[2:])

        print(f'Extracted frequencies for {self.engine_mol.name} from Gaussian log file.')
        return array(freqs)
