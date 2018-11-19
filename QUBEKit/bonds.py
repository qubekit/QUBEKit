#!/usr/bin/env python

# Bond and angle related functions used to optimize molecules in Psi4


def timer_func(orig_func):
    """Prints the runtime of a function when applied as a decorator (@timer_func)."""

    from time import time
    from functools import wraps

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time()
        result = orig_func(*args, **kwargs)
        t2 = time() - t1
        print('{} ran in: {} seconds'.format(orig_func.__name__, t2))

        return result
    return wrapper


def timer_logger(orig_func):
    """Logs the runtime of a function in a dated and numbered file.
    Outputs the start time, runtime, and function name and docstring.
    Run number can be changed with -log command."""

    from time import time
    from datetime import datetime
    from functools import wraps
    from os import listdir, path

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        start_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        t1 = time()

        result = orig_func(*args, **kwargs)

        time_taken = time() - t1

        files = [f for f in listdir('.') if path.isfile(f)]

        for file in files:
            if file.startswith('QUBEKit_log'):
                file_name = file

                with open(file_name, 'a+') as log_file:
                    log_file.write('{name} began at {starttime}.\n\nDocstring for {name}: {doc}\n\n'.format(name=orig_func.__name__, starttime=start_time, doc=orig_func.__doc__))
                    log_file.write('{name} finished in {runtime} seconds.\n\n--------------------------------------\n\n'.format(name=orig_func.__name__, runtime=time_taken))

        return result
    return wrapper


# Added to helpers
def config_loader(config_name='default_config'):
    """Sets up the desired global parameters from the config_file input.
    Allows different config settings for different projects, simply change the input config_name."""

    from importlib import import_module

    config = import_module('configs.{}'.format(config_name))

    return [config.qm, config.fitting, config.paths]


# Added to helpers
@timer_logger
def read_pdb(input_file):
    """Reads the ATOM or HETATM file to individual pdb."""

    from re import sub

    with open(input_file, 'r') as pdb:

        lines = pdb.readlines()
        molecule = []

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                element = str(line[76:78])
                # If the element column is missing from the pdb, extract the element from the name.
                if not element:
                    element = str(line.split()[2])[:-1]
                    element = sub('[0-9]+', '', element)
                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54])])

    return molecule


# Added to engines
@timer_logger
def pdb_to_psi4_geo(input_file, molecule, charge, multiplicity, basis, theory):
    """Converts to psi4in format to be used for geometric"""

    molecule_name = input_file[:-4]

    with open(molecule_name + '.psi4in', 'w+') as file:
        file.write('molecule {} {{\n {} {} \n'.format(molecule_name, charge, multiplicity))

        for i in range(len(molecule)):
            file.write(' {}    {: .6f}  {: .6f}  {: .6f}\n'.format(molecule[i][0], float(molecule[i][1]),
                                                                   float(molecule[i][2]), float(molecule[i][3])))
        file.write("}}\nset basis {}\ngradient('{}')".format(basis, theory.lower()))


# Added to engines
@timer_logger
def pdb_to_psi4(input_file, molecule, charge, multiplicity, basis, theory, memory):
    """Converts to psi4 format to be run in psi4 without using geometric"""

    molecule_name = input_file[:-4]

    with open('input.dat', 'w+') as file:
        file.write('memory {} GB\n\nmolecule {} {{\n{} {} \n'.format(memory, molecule_name, charge, multiplicity))

        for i in range(len(molecule)):
            file.write('  {}    {: .6f}  {: .6f}  {: .6f} \n'.format(molecule[i][0], float(molecule[i][1]),
                                                                    float(molecule[i][2]), float(molecule[i][3])))

        file.write("}}\n\nset basis {0}\noptimize('{1}')\nenergy, wfn = frequency('{1}', return_wfn=True)\n".format(basis, theory.lower()))
        file.write('set hessian_write on\nwfn.hessian().print_out()\n')

        # file.write('set pcm true\nset pcm_scf_type total')
        # file.write('\n\npcm = {{')
        # file.write('\n    units = Angstrom\n    Medium {{\n    SolverType = IEFPCM\n    Solvent = Chloroform\n    }}')
        # file.write('\n    Cavity {{\n    RadiiSet = UFF\n    Type = GePol\n    Scaling = False\n    Area = 0.3\n    Mode = Implicit')
        # file.write('\n    }}\n}}')


# Added to engines
@timer_logger
def input_psi4(input_file, opt_molecule, charge, multiplicity, basis, theory, memory, input_option=''):
    """Generates the name_freq.dat file in the correct format for psi4.
    If charges are required, this function will generate the cube files needed."""

    molecule_name = input_file[:-4]

    with open('{}_charge.dat'.format(molecule_name), 'w+') as file:
        file.write('memory {} GB\n\nmolecule {} {{\n {} {}\n'.format(memory, molecule_name, charge, multiplicity))

        for i in range(len(opt_molecule)):
            file.write(' {}    {: .6f}  {: .6f}  {: .6f}\n'.format(opt_molecule[i][0], float(opt_molecule[i][1]),
                                                                   float(opt_molecule[i][2]), float(opt_molecule[i][3])))

        if input_option == 'c':

            file.write("}}\n\nset basis {}\n".format(basis))
            file.write("set cubeprop_tasks ['density']\n")

            # TODO Handle overage correctly (should be dependent on the size of the molecule).
            # Methane:  12.0
            # Methanol: 17.0
            # Ethane:   16.0
            # Benzene:  24.0
            # Acetone:  20.0
            print('Calculating overage for psi4 and chargemol.')
            overage = get_overage(molecule_name)
            file.write("set CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n".format(overage))
            file.write("set CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n")
            file.write("grad, wfn = gradient('{}', return_wfn=True)\ncubeprop(wfn)".format(theory.lower()))

        elif input_option == 't':

            # TODO handle torsions properly.
            pass

        # No charges/torsions required:
        else:

            file.write("}}\n\nset basis {}\nenergy, wfn = frequency('{}', return_wfn=True)\n".format(basis, theory.lower()))
            file.write('set hessian_write on\nwfn.hessian().print_out()\n')


@timer_logger
def get_overage(molecule_name):
    """Uses the molecule's atom coordinates to quickly calculate an approximately appropriate-sized cube to enclose it.
    This is done by taking the extremal coordinates and finding the distance between their individual x,y,z points.

    There are exact algorithms for this sort of problem (convex hull problem) however they are very slow in 3D.
    As such, this serves as a reasonable approximation given there is a decent margin for error.

    Issues may arise for "odd" molecules such as if they are highly charged,
        or their shape deviates heavily from a cube shape (long chains).

    Function assumes molecule is roughly centred around (0, 0, 0) so that the box spans from large negative coords to
    large positive coords.

    *** N.B. ***
    This method has an R^2 of around 0.92 for the correlation between calculated cube size and ideal overage.
    This may be insufficient for some molecules.
    """

    # TODO Use sets rather than lists.

    with open('{}.pdb'.format(molecule_name), 'r') as file:

        coords = []
        lines = file.readlines()
        for line in lines:
            if 'HETATM' in line:
                coord = [float(line.split()[5]), float(line.split()[6]), float(line.split()[7])]
                coords.append(coord)

    # Finds the matrix transpose:
    # Transform lists of [x0, y0, z0], ... to lists of [x0, x1, ...], [y0, y1, ... ], ...
    coords = list(map(list, zip(*coords)))

    max_x, max_y, max_z = max(coords[0]), max(coords[1]), max(coords[2])
    min_x, min_y, min_z = min(coords[0]), min(coords[1]), min(coords[2])

    sq_distance = (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2

    overage = 2.988 * (sq_distance ** 0.5) + 4.678

    return overage


def modified_seminario():
    """"""

    pass


# Added to engines
@timer_logger
def pdb_to_g09(input_file, molecule, charge, multiplicity, basis, theory, processors, memory):
    """Creates g09 optimisation and frequency input files."""

    molecule_name = input_file[:-4]
    with open('{}.com'.format(molecule_name), 'w+') as file:
        file.write("%%Mem={}GB\n%%NProcShared={}\n%%Chk=lig\n# {}/{} SCF=XQC Opt=tight freq\n\nligand\n\n{} {}".format(memory, processors, theory, basis, charge, multiplicity))

        for i in range(len(molecule)):
            file.write('  {}    {: .6f}  {: .6f}  {: .6f} \n\n\n\n\n'.format(molecule[i][0], float(molecule[i][1]),
                                                                            float(molecule[i][2]), float(molecule[i][3])))


# Added to engines
@timer_logger
def extract_hess_psi4(molecule):
    """Obtains the Hessian matrix from the .hess file."""

    # TODO rewrite to not use glob module and handle exceptions properly.

    import glob
    from numpy import array, reshape

    try:
        for hessian in glob.glob("*.hess"):
            print('Hessian found')

            hess_raw = []

            with open(hessian, 'r') as hess:
                lines = hess.readlines()
                for line in lines[1:]:
                    for i in range(3):
                        hess_raw.append(float(line.split()[i]))

            hess_raw = array(hess_raw)

            # Change from Hartree/Bohr to kcal/mol /ang
            hess_form = reshape(hess_raw, (3 * len(molecule), 3 * len(molecule))) * 627.509391 / (0.529 ** 2)

            return hess_form
    except:
        raise Exception('Hessian not found.')


# Added to engines
@timer_logger
def extract_hess_psi4_geo(molecule):
    """Parses the Hessian from the B3LYP_output.dat file (from psi4) into a numpy array.
    molecule is a numpy array of size N x N"""

    from numpy import array

    hess_size = 3 * len(molecule)

    with open('B3LYP_output.dat', 'r') as file:
        lines = file.readlines()
        for count, line in enumerate(lines):
            if '## Hessian' in line:
                # Set the start of the hessian to the row of the first value.
                start_of_hess = count + 5
                # Set the end of the Hessian to the row of the last value (+1 to be sure).
                end_of_hess = start_of_hess + hess_size * (hess_size // 5 + 1) + (hess_size // 5 - 1) * 4 + 1
                hess_vals = []
                for file_line in lines[start_of_hess:end_of_hess]:
                    # Compile lists of the 5 Hessian floats for each row.
                    # Number of floats in last row may be less than 5.
                    # Only the actual floats are added, not the separating numbers.
                    row_vals = [float(val) for val in file_line.split() if len(val) > 5]
                    hess_vals.append(row_vals)

                    # Removes blank list entries.
                    # hess_vals = [elem for elem in hess_vals if elem] ?
                    hess_vals = [elem for elem in hess_vals if elem != []]

                    # Currently, every list within hess_vals is a list of 5 floats.
                    # The lists need to be concatenated so that there are n=hess_size lists of m=hess_size length.
                    # This will give a matrix of size n x m as desired.
                reshaped = []
                for i in range(hess_size):
                    row = []
                    for j in range(hess_size // 5 + 1):
                        row += hess_vals[i + j * hess_size]
                    reshaped.append(row)
                # Units conversion.
                hess_matrix = array(reshaped) * 627.509391 / (0.529 ** 2)
                return hess_matrix


# Added to engines
@timer_logger
def extract_all_modes_psi4(molecule):
    """Takes the final optimised structure from the B3LYP_output.dat file."""

    from numpy import array

    # Find "post-proj  all modes"
    # Jump to first value, ignoring text.
    # Move through data, adding it to a list
    # continue onto next line.
    # Repeat until the following line is known to be empty.

    with open('B3LYP_output.dat', 'r') as file:
        lines = file.readlines()
        for count, line in enumerate(lines):
            if "post-proj  all modes" in line:
                start_of_vals = count

                # Barring the first (and sometimes last) line, dat file has 6 values per row.
                end_of_vals = count + (3 * len(molecule)) // 6

                structures = lines[start_of_vals][24:].replace("'", "").split()
                structures = structures[6:]

                for row in range(1, end_of_vals - start_of_vals):
                    # Remove double strings and weird formatting.
                    structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

                all_modes = [float(val) for val in structures]

                return array(all_modes)


# Added to engines
@timer_logger
def extract_final_opt_struct_psi4(molecule):
    """Takes the final optimised structure from the B3LYP_output.dat file."""

    from numpy import array

    # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
    # Reverse the list
    # from the start of this list, jump down to the first atom and set this as the start point
    # Split the row into 4 columns: centre, x, y, z.
    # Add each row to a matrix.
    # Return the matrix.

    with open('B3LYP_output.dat', 'r') as file:
        lines = file.readlines()
        geo_pos_list = []       # All the lines containing '==> Geometry'.
        for count, line in enumerate(lines):
            if "==> Geometry" in line:
                geo_pos_list.append(count)

        start_of_vals = geo_pos_list[-1] + 9    # Set the start as the last instance of '==> Geometry'.

        f_opt_struct = []

        for row in range(len(molecule)):

            # Append the first 4 columns of each row, converting to float as necessary.
            struct_row = [lines[start_of_vals + row].split()[0]]
            for indx in range(1, 4):
                struct_row.append(float(lines[start_of_vals + row].split()[indx]))

            f_opt_struct.append(struct_row)

    return array(f_opt_struct)


# Added to engines
@timer_logger
def get_molecule_from_psi4(loc=''):
    """Gets the optimised structure from psi4 ready to be used for the frequency.
    Option to change the location of the file for testing purposes."""

    opt_molecule = []
    write = False

    with open(loc + 'opt.xyz', 'r') as opt:
        lines = opt.readlines()
        for line in lines:
            if 'Iteration' in line:
                print('Optimisation converged at iteration {} with final energy {}'.format(int(line.split()[1]),
                                                                                           float(line.split()[3])))
                write = True

            elif write:
                opt_molecule.append([line.split()[0], float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])

    return opt_molecule


def get_molecule_from_g09(loc=''):
    """"""

    pass
