#!/usr/bin/env python


# TODO Symmetry checks.
# TODO Check if Hydrogens are polar.


def lj_extract_params(ddec_version=6):
    """Extract the useful information from the DDEC xyz files.
    Prepare this information for the Lennard-Jones coefficient calculations."""

    # Get number of atoms from start of ddec file.

    # Extract atom types and numbers
    # Extract charges
    # Extract dipoles
    # Extract volumes (from other file)

    # Ensure total charge ~= net charge

    # return info for the molecule as a list of lists.

    if ddec_version == 6:
        net_charge_file_name = 'DDEC6_even_tempered_net_atomic_charges.xyz'

    elif ddec_version == 3:
        net_charge_file_name = 'DDEC3_net_atomic_charges.xyz'

    else:
        raise ValueError('Invalid or unsupported DDEC version.')

    with open(net_charge_file_name, 'r+') as charge_file:

        # Extract all charges, dipoles and volumes to an array for each atom in the file.

        lines = charge_file.readlines()

        # Find number of atoms
        atom_total = int(lines[0])

        for count, row in enumerate(lines):

            if 'The following XYZ' in row:

                start_pos = count + 2
                break

        molecule_data = []

        for line in lines[start_pos:start_pos + atom_total]:

            # Append the atom number and type, coords, charge, dipoles:
            # ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x dipole', 'y dipole', 'z dipole']
            atom_string_list = line.split()
            # Append all the float values first.
            atom_data = atom_string_list[2:9]
            atom_data = [float(datum) for datum in atom_data]

            # Prepend the first two values (atom_type = str, atom_number = int)
            atom_data.insert(0, atom_string_list[1])
            atom_data.insert(0, int(atom_string_list[0]))

            molecule_data.append(atom_data)

    r_cubed_file_name = 'DDEC_atomic_Rcubed_moments.xyz'

    with open(r_cubed_file_name, 'r+') as vol_file:

        lines = vol_file.readlines()

        vols = []

        for line in lines[2:atom_total + 2]:

            vol = float(line.split()[-1])
            vols.append(vol)

        for count, atom in enumerate(molecule_data):
            atom.append(vols[count])

        # Ensure total charge is near to integer value:
        total_charge = 0
        for atom in molecule_data:
            total_charge += atom[5]

        # If not 0 < total_charge << 1: you've a problem.
        if round(total_charge) - total_charge > 0.00001:
            raise ValueError('Total charge is not close enough to integer value.')

        return molecule_data


def lj_calc_coefficients(ddec_version=6):
    """Use the atom in molecule parameters from l_j_extract_params to calculate the coefficients
    of the Lennard Jones Potential.
    Exact calculations are described in full and truncated form in the comments of the function."""

    # Calculate sigma and epsilon according to paper calcs
    # Calculations from paper have been combined and simplified for faster computation.

    molecule = lj_extract_params(ddec_version)

    # 'elem' : [vfree, bfree, rfree]

    # TODO Is bfree for Carbon 46.5 or 46.6? Differs across papers.
    # TODO Test values need to be removed or changed later (Zn, Au)
    elem_dict = {
        'H': [7.6, 6.5, 1.64],
        'C': [34.4, 46.5, 2.08],
        'N': [25.9, 24.2, 1.72],
        'O': [22.1, 15.6, 1.60],
        'F': [18.2, 9.5, 1.58],
        'S': [75.2, 134.0, 2.00],
        'Cl': [65.1, 94.6, 1.88],
        'Br': [95.7, 162.0, 1.96],
        'Zn': [1, 1, 1],
        'Au': [1, 1, 1]
    }

    # Conversion from Ha.Bohr ** 6 to kcal / (mol * Ang ** 6):
    kcal_ang = 13.7792544

    sigmas = []
    epsilons = []

    for atom in molecule:

        # fac = atom volume / vfree
        fac = atom[-1] / elem_dict[f'{atom[1]}'][0]

        # sigma = 2 ** (5 / 6) * fac ** (1 / 3) * rfree
        sigma = 1.781797436 * (fac ** (1 / 3)) * elem_dict[f'{atom[1]}'][2]

        # epsilon = bfree / ((2 ** 7) * (rfree ** 6))
        epsilon = elem_dict[f'{atom[1]}'][1] / (128 * (elem_dict[f'{atom[1]}'][2]) ** 6)
        epsilon *= kcal_ang

        ##############

        # Longer method than necessary but more closely follows paper:

        # r_aim = rfree * ((vol / vfree) ** (1 / 3))
        # r_aim = elem_dict[f'{atom[1]}'][2] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** (1 / 3))

        # b_i = bfree * ((vol / vfree) ** 2)
        # b_i = elem_dict[f'{atom[1]}'][1] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** 2)

        # a_i = 0.5 * b_i * ((2 * r_aim) ** 6)

        # sigma = (a_i / b_i) ** (1 / 6)

        # epsilon = (b_i ** 2) / (4 * a_i)

        ##############

        sigmas.append(sigma)
        epsilons.append(epsilon)

    return sigmas, epsilons
