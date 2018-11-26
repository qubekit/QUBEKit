#!/usr/bin/env python


from QUBEKit.helpers import config_loader, get_overage
from QUBEKit.decorators import for_all_methods, timer_logger

from subprocess import call as sub_call
from os import environ


class Engines:
    """Engines superclass containing core information that all other engines (PSI4, Gaussian etc) will have.
    Provides atoms' coordinates with name tags for each atom and entire molecule.
    Also gives all configs from the appropriate config file.
    """

    def __init__(self, molecule, config_dict):
        # Obtains the molecule name and a list of elements in the molecule with their respective coordinates.
        # self.molecule_name, self.molecule = molecule.name, molecule.molecule
        self.engine_mol = molecule
        self.charge = config_dict['charge']
        self.multiplicity = config_dict['multiplicity']
        self.geometric = config_dict['geometric']
        self.solvent = config_dict['solvent']
        self.ddec_version = config_dict['ddec version']
        # Load the configs using the config_file name.
        confs = config_loader(config_dict['config'])
        self.qm, self.fitting, self.paths = confs

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'


@for_all_methods(timer_logger)
class PSI4(Engines):
    """Psi4 class (child of Engines).
    Used to extract optimised structures, Hessians, frequencies, etc.
    """

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

    def hessian(self):
        """Parses the Hessian from the B3LYP_output.dat file (from psi4) into a numpy array.
        molecule is a numpy array of size N x N
        """

        from numpy import array

        hess_size = len(self.engine_mol.molecule) * 3

        # B3LYP_output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:

            lines = file.readlines()

            for count, line in enumerate(lines):

                if '## Hessian' in line:
                    # Set the start of the hessian to the row of the first value.
                    start_of_hess = count + 5

                    # Check if the hessian continues over onto more lines (i.e. if hess_size is not divisible by 5)
                    if hess_size % 5 == 0:
                        extra = 0
                    else:
                        extra = 1
                    # length_of_hess: #of cols * length of each col
                    #                +#of cols - 1 * #blank lines per row of hess_vals
                    #                +#blank lines per row of hess_vals if the hess_size continues over onto more lines.
                    length_of_hess = (hess_size // 5) * hess_size + (hess_size // 5 - 1) * 3 + extra * (3 + hess_size)

                    end_of_hess = start_of_hess + length_of_hess

                    hess_vals = []

                    for file_line in lines[start_of_hess:end_of_hess]:
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

                    # Units conversion.
                    hess_matrix = array(reshaped) * 627.509391 / (0.529 ** 2)

                    # Check matrix is symmetric to within some error.
                    error = 0.00001

                    for i in range(len(hess_matrix)):
                        for j in range(len(hess_matrix)):
                            if abs(hess_matrix[i, j] - hess_matrix[j, i]) > error:
                                raise Exception('Hessian is not symmetric.')

                    print(f'Extracted Hessian for {self.engine_mol.name} from psi4 output.')
                    self.engine_mol.hessian = hess_matrix

                    return self.engine_mol

    def optimised_structure(self):
        """Parses the final optimised structure from the B3LYP_output.dat file (from psi4) to a numpy array.
        """

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # B3LYP_output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if "==> Geometry" in line:
                    geo_pos_list.append(count)

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            f_opt_struct = []

            for row in range(len(self.engine_mol.molecule)):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = [lines[start_of_vals + row].split()[0]]
                for indx in range(1, 4):
                    struct_row.append(float(lines[start_of_vals + row].split()[indx]))

                f_opt_struct.append(struct_row)

        print(f'Extracted optimised structure for {self.engine_mol.name} from psi4 output.')
        self.engine_mol.QMoptimized = f_opt_struct

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
            for i in range(len(molecule)):
                input_file.write(' {}    {: .10f}  {: .10f}  {: .10f} \n'.format(molecule[i][0], float(molecule[i][1]),
                                                                                 float(molecule[i][2]),
                                                                                 float(molecule[i][3])))
            input_file.write(' units angstrom\n no_reorient\n}}\n\nset {{\n basis {}\n'.format(self.qm['basis']))

            if optimize:
                print('Writing Psi4 optimization input')
                setters += ' ng_convergence {}\n GEOM_MAXITER {}\n'.format(self.qm['convergence'], self.qm['iterations'])
                tasks += "\noptimize('{}')".format(self.qm['theory'].lower())

            if hessian:
                print('Writing Psi4 hessian calculation input')
                setters += ' hessian_write on\n'
                tasks += "\nenergy, wfn = frequency('{}', return_wfn=True)".format(self.qm['theory'].lower())
                tasks += '\nwfn.hessian().print_out()\n\n'

            if density:
                print('Writing Psi4 Density calculation input')
                setters += " cubeprop_tasks ['density']\n"
                # TODO Handle overage correctly (should be dependent on the size of the molecule)
                # See helpers.get_overage for info.

                # print('Calculating overage for psi4 and chargemol.')
                overage = get_overage(self.engine_mol.name)
                setters += " CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n".format(overage)
                setters += " CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n"
                tasks += "grad, wfn = gradient('{}', return_wfn=True)\ncubeprop(wfn)".format(self.qm['theory'].lower())

            # TODO check the input settings and compare with g09
            if self.solvent:
                print('Setting pcm parameters.')
                input_file.write('\n\nset pcm true\nset pcm_scf_type total')
                input_file.write('\n\npcm = {')
                input_file.write(
                    '\n    units = Angstrom\n    Medium {\n    SolverType = IEFPCM\n    Solvent = Chloroform\n    }')
                input_file.write(
                    '\n    Cavity {\n    RadiiSet = UFF\n    Type = GePol\n    Scaling = False\n    Area = 0.3\n    '
                    'Mode = Implicit')
                input_file.write('\n    }\n}')

            setters += '}\n'
            if threads:
                setters += f'set_num_threads({self.qm["threads"]})\n'
            input_file.write(setters)
            input_file.write(tasks)

        sub_call(f'psi4 input.dat -n {self.qm["threads"]}', shell=True)

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        from numpy import array

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # B3LYP_output.dat is the psi4 output file.
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
        """"Write the psi4 style input file to get the gradient for geometric
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
            for i in range(len(molecule)):
                file.write('  {}    {: .10f}  {: .10f}  {: .10f}\n'.format(molecule[i][0], float(molecule[i][1]),
                                                                           float(molecule[i][2]),
                                                                           float(molecule[i][3])))

            file.write("units angstrom\n no_reorient\n}}\nset basis {}\n".format(self.qm['basis']))
            if threads:
                file.write('set_num_threads({})'.format(self.qm['threads']))
            file.write("\n\ngradient('{}')\n".format(self.qm['theory']))

        if run:
            with open('log.txt', 'w+') as log:
                sub_call(f'geometric-optimize --psi4 {self.engine_mol.name}.psi4in --nt {self.qm["threads"]}',
                         shell=True, stdout=log)
        else:
            print('Geometric psi4 optimise file written.')


@for_all_methods(timer_logger)
class Chargemol(Engines):

    def __init__(self, molecule, config_file):

        super().__init__(molecule, config_file)

    def generate_input(self):
        """Given a DDEC version (from the defaults), this function writes the job file for chargemol."""

        if self.ddec_version != 6 or self.ddec_version != 3:
            print('Invalid or unsupported DDEC version given, running with default version 6.')
            self.ddec_version = 6

        # Write the charges job file.
        with open('job_control.txt', 'w+') as charge_file:

            charge_file.write(f'<input filename>\n{self.engine_mol.name}.wfx\n</input filename>')

            charge_file.write('\n\n<net charge>\n0.0\n</net charge>')

            charge_file.write('\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false.')
            charge_file.write('\n</periodicity along A, B and C vectors>')

            charge_file.write(f'\n\n<atomic densities directory complete path>\n{self.paths["chargemol"]}/atomic_densities/')
            charge_file.write('\n</atomic densities directory complete path>')

            charge_file.write(f'\n\n<charge type>\nDDEC{self.ddec_version}\n</charge type>')

            charge_file.write('\n\n<compute BOs>\n.true.\n</compute BOs>')

        # sub_call(f'psi4 input.dat -n {self.qm["threads"]}', shell=True)
        # sub_call('mv Dt.cube total_density.cube', shell=True)

        sub_call(f'{self.paths["chargemol"]} /chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_serial job_control.txt',
                 shell=True)

    def extract_charges(self):
        """Extract the charge data from the chargemol execution."""

        pass


@for_all_methods(timer_logger)
class Gaussian(Engines):

    def __init__(self, molecule, config_dict):

        super().__init__(molecule, config_dict)

    def generate_input(self, QM=False, MM=False, optimize=False, hessian=False, density=False):

        if QM:
            molecule = self.engine_mol.QMoptimized
        elif MM:
            molecule = self.engine_mol.MMoptimized
        else:
            molecule = self.engine_mol.molecule

        with open(f'gj_{self.engine_mol.name}', 'w+') as input_file:

            input_file.write(f'%Mem={self.qm["memory"]}GB\n%NProcShared={self.qm["threads"]}\n%Chk=lig\n')

            commands = f'# {self.qm["theory"]}/{self.qm["basis"]} SCF=XQC '

            if density:
                commands += 'density=current OUTPUT=WFX '

            if optimize:
                commands += 'opt '

            if hessian:
                commands += 'freq '

            commands += f'\n\n{self.engine_mol.name}\n\n'
            commands += f'{self.charge} {self.multiplicity}\n'

            input_file.write(commands)

            for atom in range(len(molecule)):
                input_file.write('{} {: .3f} {: .3f} {: .3f}\n'.format(molecule[atom][0], float(molecule[atom][1]),
                                                                       float(molecule[atom][2]), float(molecule[atom][3])))
            if density:
                input_file.write(f'\n{self.engine_mol.name}.wfx')

            input_file.write('\n\n\n\n\n')

        if 'g09' in environ:
            sub_call(f'g09 < gj_{self.engine_mol.name} > gj_{self.engine_mol.name}.log', shell=True)
        else:
            raise FileNotFoundError('cannot run Gaussian')
