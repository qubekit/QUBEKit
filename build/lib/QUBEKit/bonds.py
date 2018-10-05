#!/usr/bin/env python

# Bond and angle related functions used to optimize molecules in Psi4


def config_setup():
    """Set up the config file with the correct global parameters to be used throughout."""

    from os import path, environ
    import configparser

    config = configparser.RawConfigParser()
    # Look for config file
    loc = environ.get("QUBEKit")

    # try:
    #     with open(path.join(loc, "QUBEKit/new_config.ini")) as source:
    #         config.readfp( source )
    #         theory = str(config['qm']['theory'])

    with open('default_config.py', 'r') as source:
        config.readfp(source)

    # except:
    #     with open(path.join(loc, "QUBEKit/default_config.py")) as source:
    #         config.readfp( source )
    #         theory = str(config['qm']['theory'])

    theory = str(config['qm']['theory'])
    basis = str(config['qm']['basis'])
    vib_scaling = float(config['qm']['vib_scaling'])
    processors = int(config['qm']['processors'])
    memory = int(config['qm']['memory'])

    dihstart = int(config['fitting']['dihstart'])
    increment = int(config['fitting']['increment'])
    numscan = int(config['fitting']['numscan'])
    T_weight = str(config['fitting']['T_weight'])
    new_dihnum = int(config['fitting']['new_dihnum'])
    Q_file = config['fitting']['Q_file']
    tor_limit = int(config['fitting']['tor_limit'])
    div_index = int(config['fitting']['div_index'])

    chargemol = str(config['paths']['chargemol'])

    return theory, basis, vib_scaling, processors, memory, dihstart, increment, numscan, T_weight, new_dihnum, Q_file, tor_limit, div_index, chargemol


def read_pdb(input_file):
    """Reads the ATOM or HETATM file to individual pdb."""

    from re import sub

    with open(input_file, 'r') as pdb:

        lines = pdb.readlines()
        molecule = []

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                element = str(line[76:78])
                # If the element column is missing from the pdb extract the element from the name.
                if len(element) < 1:
                    element = str(line.split()[2])[:-1]
                    element = sub('[0-9]+', '', element)
                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54])])

    return molecule


def pdb_to_psi4_geo(input_file, molecule, charge, multiplicity, basis, theory):
    """Converts to psi4in format to be used for geometric"""

    from os import mkdir, system

    molecule_name = input_file[:-4]
    out = open(molecule_name+'.psi4in', 'w+')
    out.write('molecule {} {{\n {} {} \n'.format(molecule_name, charge, multiplicity))

    for i in range(len(molecule)):
        out.write('  {}    {: .6f}  {: .6f}  {: .6f}\n'.format(molecule[i][0], float(molecule[i][1]), float(molecule[i][2]), float(molecule[i][3])))
    out.write("}}\nset basis {}\ngradient('{}')".format(basis, theory.lower()))
    out.close()

    try:
        mkdir('BONDS')
    except:
        pass

    system('mv {}.psi4in BONDS'.format(molecule_name))
    view = input('psi4 input made and moved to BONDS\nwould you like to view the file\n>')

    if view.lower() in ('y' or 'yes'):
        psi4in = open('BONDS/{}.psi4in'.format(molecule_name), 'r').read()
        print(psi4in)


# Convert to psi4 format to be run in psi4 without geometric
def pdb_to_psi4(input_file, molecule, charge, multiplicity, basis, theory, memory):
    """Converts to psi4 format to be run in psi4 without using geometric"""

    from os import mkdir, system

    molecule_name = input_file[:-4]
    out = open('input.dat', 'w+')
    out.write("memory {} GB\n\nmolecule {} {\n{} {} \n".format(memory, molecule_name, charge, multiplicity))

    for i in range(len(molecule)):
        out.write('  {}    {: .6f}  {: .6f}  {: .6f} \n'.format(molecule[i][0], float(molecule[i][1]),
                                                               float(molecule[i][2]), float(molecule[i][3])))

    out.write("}}\n\nset basis {0}\noptimize('{1}')\nenergy, wfn = frequency('{1}', return_wfn=True)\n".format(basis, theory.lower()))
    out.write('set hessian_write on\nwfn.hessian().print_out()\n')
    out.close()

    try:
        mkdir('BONDS')
    except:
        pass

    system('mv input.dat BONDS')
    view = input('psi4 input made and moved to BONDS\nwould you like to view the file\n>')

    if view.lower() in ('y' or 'yes'):
        psi4in = open('BONDS/input.dat', 'r').read()
        print(psi4in)


def input_psi4(input_file, opt_molecule, charge, multiplicity, basis, theory, memory, input_option=''):
    """Generates the name_freq.dat file in the correct format for psi4.
    If charges are required, this function will generate the cube files needed."""

    molecule_name = input_file[:-4]

    out = open('{}_freq.dat'.format(molecule_name), 'w+')
    out.write('memory {} GB \n\nmolecule {} {{ \n {} {} \n'.format(memory, molecule_name, charge, multiplicity))

    for i in range(len(opt_molecule)):
        out.write(' {}    {: .6f}  {: .6f}  {: .6f} \n'.format(opt_molecule[i][0], float(opt_molecule[i][1]),
                                                  float(opt_molecule[i][2]), float(opt_molecule[i][3])))

    if input_option in ('c' or 'charge' or 'charges'):

        out.write("}} \n\nset basis {} \nenergy, wfn = energy('{}', return_wfn=True) \n".format(basis, theory.lower()))
        out.write("set cubeprop_tasks ['density']\ncubeprop(wfn)\n")
        out.write("energy, wfn = frequency('{}', return_wfn=True) \n".format(theory.lower()))
        out.write('set hessian_write on \nwfn.hessian().print_out() \n')

    elif input_option in ('t' or 'torsion' or 'torsions'):
        # TODO handle torsions properly.
        pass

    # No charges/torsions required:
    else:

        out.write("}} \n\nset basis {} \nenergy, wfn = frequency('{}', return_wfn=True) \n".format(basis, theory.lower()))
        out.write('set hessian_write on \nwfn.hessian().print_out() \n')

    out.close()


# Function to use the modified seminario method
def modified_seminario():
    pass


def pdb_to_g09(input_file, molecule, charge, multiplicity, basis, theory, processors, memory):
    """Creates g09 optimisation and frequency input files."""

    from os import mkdir, system

    molecule_name = input_file[:-4]
    out = open('{}.com'.format(molecule_name), 'w+')
    out.write("%%Mem={}GB\n%%NProcShared={}\n%%Chk=lig\n# {}/{} SCF=XQC Opt=tight freq\n\nligand\n\n{} {}".format(memory, processors, theory, basis, charge, multiplicity))

    for i in range(len(molecule)):
        out.write('  {}    {: .6f}  {: .6f}  {: .6f} \n\n\n\n\n'.format(molecule[i][0], float(molecule[i][1]), float(molecule[i][2]), float(molecule[i][3])))
    out.close()

    try:
        mkdir('BONDS')
    except:
        pass

    system('mv {}.com BONDS'.format(molecule_name))
    view = input('gaussian09 input made and moved to BONDS \nwould you like to view the file? \n>')
    if view.lower() in ('y' or 'yes'):
        g09 = open('BONDS/{}.com'.format(molecule_name), 'r').read()
        print(g09)


def extract_hess_psi4(molecule):
    """Obtains the Hessian matrix from the psi4 output."""

    import glob
    from numpy import array, reshape
    from sys import exit

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
        exit('Hessian missing!')


def extract_hessian_psi4(molecule):
    """Parses the Hessian from the output.dat file into a numpy array.
    molecule is a numpy array of size N x N"""

    from numpy import array

    hess_size = 3 * len(molecule)

    with open('output.dat', 'r') as file:
        lines = file.readlines()
        for count, line in enumerate(lines):
            if '## Hessian' in line:
                # Set the start of the hessian to the row of the first value.
                start_of_hess = count + 5
                # Set the end of the Hessian to the row of the last value (+1 to be sure).
                end_of_hess = start_of_hess + hess_size * (hess_size // 5 + 1) + (hess_size // 5 - 1) * 4 + 1
                hess_vals = []
                for file_line in lines[start_of_hess:end_of_hess]:
                    # Compile lists of the 5 Hessian floats for each row
                    # Number of floats in last row may be less than 5
                    # Only the actual floats are added, not the separating numbers
                    row_vals = [float(val) for val in file_line.split() if len(val) > 5]
                    hess_vals.append(row_vals)
                    # Removes blank list entries
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
                hess_matrix = array(reshaped) * 627.509391 / (0.529 ** 2)
    return hess_matrix


def extract_all_modes_psi4(molecule):
    """Takes the final optimised structure from the output.dat file."""

    from numpy import array

    # Find "post-proj  all modes"
    # Jump to first value, ignoring text.
    # Move through data, adding it to a list
    # continue onto next line.
    # Repeat until the following line is known to be empty.

    with open('output.dat', 'r') as file:
        lines = file.readlines()
        for count, line in enumerate(lines):
            if "post-proj  all modes" in line:
                start_of_vals = count

                # Barring the first (and sometimes last) line, dat file has 6 values per row.
                end_of_vals = count + (3 * len(molecule)) // 6

                structures = lines[start_of_vals][24:].replace("'", "").split()
                structures = structures[6:]

                for row in range(1, end_of_vals - start_of_vals):
                    structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

                all_modes = [float(val) for val in structures]

    return array(all_modes)


def extract_final_opt_struct_psi4(molecule):
    """Takes the final optimised structure from the output.dat file."""

    from numpy import array

    # Run through the file backwards until '==> Geometry' is found.
    # This is the initial start point.
    # Jump down to the first atom and set this as the start point
    # Split the row into 4 columns: centre, x, y, z.
    # Add each row to a matrix.
    # Return the matrix.

    with open('output.dat', 'r') as file:
        lines = file.readlines()
        geo_pos_list = []       # All the lines containing '==> Geometry'.
        for count, line in enumerate(lines):
            if "==> Geometry" in line:
                geo_pos_list.append(count)

        start_of_vals = geo_pos_list[-1] + 9    # Set the start from the last instance of '==> Geometry'.

        f_opt_struct = []
        print(len(molecule))

        for row in range(len(molecule)):

            # Append the first 4 columns of each row, converting to float as necessary.
            struct_row = [lines[start_of_vals + row].split()[0]]
            struct_row.append(float(lines[start_of_vals + row].split()[1]))
            struct_row.append(float(lines[start_of_vals + row].split()[2]))
            struct_row.append(float(lines[start_of_vals + row].split()[3]))

            f_opt_struct.append(struct_row)

    return array(f_opt_struct)


def get_molecule_from_psi4(loc=''):
    """Gets the optimised structure from psi4 ready to be used for the frequency.
    Option to change the location of the file for testing purposes."""

    opt_molecule = []
    write = False

    with open(loc + 'opt.xyz', 'r') as opt:
        lines = opt.readlines()
        for line in lines:
            if 'Iteration' in line:
                print('optimization converged at iteration {} with final energy {}'.format(int(line.split()[1]), float(line.split()[3])))
                write = True

            elif write:
                opt_molecule.append([line.split()[0], float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])

    return opt_molecule


# Function to get the optimized molecule from g09
def get_molecule_from_g09():
    pass
