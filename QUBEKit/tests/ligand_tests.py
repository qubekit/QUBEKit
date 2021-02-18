import pytest

from QUBEKit.ligand import Ligand
from QUBEKit.utils.file_handling import get_data

# acetone_data = [
#     ("name", "acetone"),
#     (
#         "topology.edges",
#         [(0, 4), (0, 5), (0, 6), (0, 1), (1, 2), (1, 3), (3, 7), (3, 8), (3, 9)],
#     ),
#     (
#         "angles",
#         [
#             (1, 0, 4),
#             (1, 0, 5),
#             (1, 0, 6),
#             (4, 0, 5),
#             (4, 0, 6),
#             (5, 0, 6),
#             (0, 1, 2),
#             (0, 1, 3),
#             (2, 1, 3),
#             (1, 3, 7),
#             (1, 3, 8),
#             (1, 3, 9),
#             (7, 3, 8),
#             (7, 3, 9),
#             (8, 3, 9),
#         ],
#     ),
#     (
#         "dihedrals",
#         {
#             (0, 1): [
#                 (4, 0, 1, 2),
#                 (4, 0, 1, 3),
#                 (5, 0, 1, 2),
#                 (5, 0, 1, 3),
#                 (6, 0, 1, 2),
#                 (6, 0, 1, 3),
#             ],
#             (1, 3): [
#                 (0, 1, 3, 7),
#                 (0, 1, 3, 8),
#                 (0, 1, 3, 9),
#                 (2, 1, 3, 7),
#                 (2, 1, 3, 8),
#                 (2, 1, 3, 9),
#             ],
#         },
#     ),
#     ("impropers", [(1, 0, 2, 3)]),
# ]


@pytest.mark.parametrize(
    "input_molecule",
    [
        pytest.param(
            (
                "acetone.pdb",
                [
                    (0, 4),
                    (0, 5),
                    (0, 6),
                    (0, 1),
                    (1, 2),
                    (1, 3),
                    (3, 7),
                    (3, 8),
                    (3, 9),
                ],
            ),
            id="acetone",
        ),
    ],
)
def test_bonds(input_molecule):
    """Make sure after reading a molecule file that we have the correct set of bonds."""
    molecule, bonds = input_molecule
    mol = Ligand(get_data(molecule))
    for bond in bonds:
        assert bond in mol.bonds


# @pytest.mark.parametrize("test_input, expected", test_cases)
# def test_mol2_reader(test_input, expected):
#     mol = Ligand("acetone.mol2")
#     mol.testing = True
#     assert getattr(mol, test_input) == expected
#
#
# @pytest.mark.parametrize("test_input, expected", test_cases)
# def test_smiles_reader(test_input, expected):
#     mol = Ligand("CC(=O)C", "acetone")
#     mol.testing = True
#     assert getattr(mol, test_input) == expected


# import os
# from shutil import copy, rmtree
# import unittest
#
#
# class TestLigands(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         """
#         Create temp working directory and copy across test files.
#         Initialise test molecule (acetone) with Ligand().
#         """
#
#         cls.files_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
#         os.mkdir('temp')
#         os.chdir('temp')
#         copy(os.path.join(cls.files_folder, 'acetone.pdb'), 'acetone.pdb')
#         cls.molecule_pdb = Ligand('acetone.pdb')
#         cls.molecule_pdb.testing = True
#         copy(os.path.join(cls.files_folder, 'acetone.mol2'), 'acetone.mol2')
#         cls.molecule_mol2 = Ligand('acetone.mol2')
#         cls.molecule_mol2.testing = True
#         os.chdir('../')
#
#     def test_pdb_reader(self):
#         """Ensure pdb has been interpreted properly."""
#
#         self.assertEqual(10, len(self.molecule_pdb.atoms))
#
#         for atom in self.molecule_pdb.coords['input']:
#             self.assertEqual(3, len(atom))
#
#         for atom in self.molecule_pdb.atoms:
#             self.assertIsNotNone(atom.atom_name)
#             self.assertIsNotNone(atom.atom_index)
#             self.assertIsNotNone(atom.atomic_mass)
#             self.assertIsNotNone(atom.atomic_symbol)
#             self.assertIsNotNone(atom.atomic_number)
#
#     def test_mol2_reader(self):
#         """Ensure mol2 has been interpreted properly."""
#
#         # Check all atoms are found
#         self.assertEqual(10, len(self.molecule_mol2.coords['input']))
#
#         # Check atom names and coords are extracted for each atom in the molecule
#         for atom in self.molecule_mol2.coords['input']:
#             self.assertEqual(3, len(atom))
#
#         for atom in self.molecule_pdb.atoms:
#             self.assertIsNotNone(atom.atom_name)
#             self.assertIsNotNone(atom.atom_index)
#             self.assertIsNotNone(atom.atomic_mass)
#             self.assertIsNotNone(atom.atomic_symbol)
#             self.assertIsNotNone(atom.atomic_number)
#
#     def test_bonds(self):
#         """Ensure graph structure has been built properly."""
#
#         bonds = [(0, 4), (0, 5), (0, 6), (0, 1), (1, 2), (1, 3), (3, 7), (3, 8), (3, 9)]
#         self.assertEqual(bonds, list(self.molecule_pdb.topology.edges))
#         self.assertEqual(bonds, list(self.molecule_mol2.topology.edges))
#
#         self.assertEqual(len(self.molecule_pdb.bond_lengths), len(list(self.molecule_pdb.topology.edges)))
#         self.assertEqual(len(self.molecule_mol2.bond_lengths), len(list(self.molecule_mol2.topology.edges)))
#
#     def test_angles(self):
#         """Ensure angles have been generated properly."""
#
#         angles = [(1, 0, 4), (1, 0, 5), (1, 0, 6), (4, 0, 5), (4, 0, 6), (5, 0, 6), (0, 1, 2), (0, 1, 3), (2, 1, 3),
#                   (1, 3, 7), (1, 3, 8), (1, 3, 9), (7, 3, 8), (7, 3, 9), (8, 3, 9)]
#
#         self.assertEqual(angles, list(self.molecule_pdb.angles))
#         self.assertEqual(angles, list(self.molecule_mol2.angles))
#
#         self.assertEqual(len(self.molecule_pdb.angles), len(self.molecule_pdb.angle_values))
#         self.assertEqual(len(self.molecule_mol2.angles), len(self.molecule_mol2.angle_values))
#
#     def test_dihedrals(self):
#         """Ensure dihedrals have been identified properly."""
#
#         dihedrals = {(0, 1): [(4, 0, 1, 2), (4, 0, 1, 3), (5, 0, 1, 2), (5, 0, 1, 3), (6, 0, 1, 2), (6, 0, 1, 3)],
#                      (1, 3): [(0, 1, 3, 7), (0, 1, 3, 8), (0, 1, 3, 9), (2, 1, 3, 7), (2, 1, 3, 8), (2, 1, 3, 9)]}
#         self.assertEqual(dihedrals, self.molecule_pdb.dihedrals)
#         self.assertEqual(dihedrals, self.molecule_mol2.dihedrals)
#
#         # check that every dihedral has a value measured
#         self.assertEqual(12, len(self.molecule_pdb.dih_phis))
#         self.assertEqual(12, len(self.molecule_mol2.dih_phis))
#
#         impropers = [(1, 0, 2, 3)]
#         self.assertEqual(impropers, self.molecule_pdb.improper_torsions)
#         self.assertEqual(impropers, self.molecule_mol2.improper_torsions)
#
#         # Check the rotatable torsions found;
#         # this ensures that methyl groups are removed from the torsion list and symmetrisation is working
#         rot = None
#         self.assertEqual(rot, self.molecule_pdb.rotatable)
#         self.assertEqual(rot, self.molecule_mol2.rotatable)
#
#     def test_smiles(self):
#         """Ensure molecule is correctly generated from a smiles string."""
#
#         molecule_smiles = Ligand('CCC', 'propane')
#
#         angles = {(1, 0, 3): 113.51815048217622, (1, 0, 4): 108.585923222101, (1, 0, 5): 106.72547221240829,
#                   (3, 0, 4): 108.67471750338844, (3, 0, 5): 109.86966536530876, (4, 0, 5): 109.3960638804494,
#                   (0, 1, 2): 112.47821537702777, (0, 1, 6): 106.25702918976113, (0, 1, 7): 113.72590402122567,
#                   (2, 1, 6): 106.3390387838715, (2, 1, 7): 111.69729882714941, (6, 1, 7): 105.65819247884409,
#                   (1, 2, 8): 108.59874810898711, (1, 2, 9): 112.19545440609062, (1, 2, 10): 111.67294842834627,
#                   (8, 2, 9): 111.33448705926884, (8, 2, 10): 107.47750840394838, (9, 2, 10): 105.46240504563437}
#
#         # loop over the angles and make sure they are almost equal to 7 d.p
#         # the dictionaries will always be made in the same order unless the RDKit ordering changes.
#         for angle, value in angles.items():
#             self.assertAlmostEqual(molecule_smiles.angle_values[angle], value)
#
#     @classmethod
#     def tearDownClass(cls):
#         """Remove the files produced during testing."""
#
#         rmtree('temp')
#
#
# if __name__ == '__main__':
#
#     unittest.main()
