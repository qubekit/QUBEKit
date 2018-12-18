#!/usr/bin/env python


def pdb_to_coord_list(pdb_file):
    """Opens a pdb file ('methane.pdb' for example) and returns the molecule name, followed by a list of the coords.
    Coords will be of the form: [['C', -1.28, 0.127, -0.003], ['C', 0.029, -0.428, -0.47] ... ]
    """

    # TODO Ensure this works with different pdb styles.

    molecule = []

    with open(pdb_file, 'r') as file:
        lines = file.readlines()
        # First line contains the molecule name as the last entry.
        molecule_name = lines[0].split()[1]

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                # Create a list containing the element's name and coords, e.g. ['C', -1.28, 0.127, -0.003]
                element_name = [line.split()[-1]]
                coords = line.split()[-6:-3]

                atom = element_name + coords
                molecule.append(atom)
        print('Read and transcribed pdb for {}'.format(molecule_name))
        return molecule_name, molecule


def config_loader(config_name='default_config'):
    """Sets up the desired global parameters from the config_file input.
    Allows different config settings for different projects, simply change the input config_name."""

    from importlib import import_module

    config = import_module('configs.{}'.format(config_name))

    return [config.qm, config.fitting, config.paths]


def get_overage(molecule):
    """Bodge."""

    overage_dict = {'methane': 12.0, 'ethane': 16.0, 'acetone': 20.0, 'benzene': 24.0, 'methanol': 17.0}
    return overage_dict[molecule]


def unpickle(pickle_jar):
    """Function to unpickle a set of ligand objects from the pickle file, and return a dictionary of liagnds
    indexed by their progress."""

    from pickle import load

    mol_states = {}
    mols = []
    # unpickle the pickle jar
    # try to load a pickle file make sure to get all objects
    with open(pickle_jar, 'rb') as jar:
        while True:
            try:
                mols.append(load(jar))
            except:
                break
    # for each object in the jar put them into a dictionary indexed by there state
    for mol in mols:
        mol_states[mol.state] = mol

    return mol_states


