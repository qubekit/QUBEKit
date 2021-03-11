#!/usr/bin/env python3
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule
from rdkit.Geometry.rdGeometry import Point3D

from QUBEKit.utils.exceptions import FileTypeError, SmartsError


class RDKit:
    """Class for controlling useful RDKit functions."""

    @staticmethod
    def mol_to_file(rdkit_mol: Chem.Mol, file_name: str) -> None:
        """
        Write the rdkit molecule to the requested file type.
        #TODO do we want multiframe support for all files? just xyz?
        """
        file_path = Path(file_name)
        if file_path.suffix == ".pdb":
            return Chem.MolToPDBFile(rdkit_mol, file_name)
        elif file_path.suffix == ".sdf" or file_path.suffix == ".mol":
            return Chem.MolToMolFile(rdkit_mol, file_name)
        elif file_path.suffix == ".xyz":
            with open(file_name, "w") as xyz:
                for i in range(rdkit_mol.GetNumConformers()):
                    xyz.write(Chem.MolToXYZBlock(rdkit_mol, confId=i))
        else:
            raise FileTypeError(
                f"The file type {file_path.suffix} is not supported please chose from xyz, pdb, mol or sdf."
            )

    @staticmethod
    def file_to_rdkit_mol(file_path: Path) -> Chem.Mol:
        """
        :param mol_input: pathlib.Path of the filename provided or the smiles string
        :param name:
        :return: RDKit molecule object generated from its file (or None if incorrect file type is provided).
        """

        # Read the file
        if file_path.suffix == ".pdb":
            mol = Chem.MolFromPDBFile(
                file_path.as_posix(), removeHs=False, sanitize=False
            )
        elif file_path.suffix == ".mol2":
            mol = Chem.MolFromMol2File(
                file_path.as_posix(), removeHs=False, sanitize=False
            )
        elif file_path.suffix == ".mol" or file_path.suffix == ".sdf":
            mol = Chem.MolFromMolFile(
                file_path.as_posix(), removeHs=False, sanitize=False, strictParsing=True
            )
        else:
            raise FileTypeError(f"The file type {file_path.suffix} is not supported.")
        # run some sanitation
        Chem.SanitizeMol(
            mol,
            (Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY ^ Chem.SANITIZE_ADJUSTHS),
        )
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
        Chem.AssignStereochemistryFrom3D(mol)

        # set the name of the input file
        mol.SetProp("_Name", file_path.stem)

        return mol

    @staticmethod
    def smiles_to_rdkit_mol(smiles_string: str, name: Optional[str] = None):
        """
        Converts smiles strings to RDKit mol object.
        :param smiles_string: The hydrogen free smiles string
        :param name: The name of the molecule this will be used when writing the pdb file
        :return: The RDKit molecule
        """

        mol = AllChem.MolFromSmiles(smiles_string)
        if name is None:
            name = input("Please enter a name for the molecule:\n>")
        mol.SetProp("_Name", name)
        mol_hydrogens = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol_hydrogens, AllChem.ETKDG())
        AllChem.SanitizeMol(mol_hydrogens)

        return mol_hydrogens

    @staticmethod
    def mm_optimise(rdkit_mol: Chem.Mol, ff="MMF") -> Chem.Mol:
        """
        Perform rough preliminary optimisation to speed up later optimisations.
        :param filename: The Path of the input file
        :param ff: The Force field to be used either MMF or UFF
        :return: The name of the optimised pdb file that is made
        """

        {"MMF": MMFFOptimizeMolecule, "UFF": UFFOptimizeMolecule}[ff](rdkit_mol)
        return rdkit_mol

    @staticmethod
    def rdkit_descriptors(rdkit_mol: Chem.Mol) -> Dict[str, float]:
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary.
        :param rdkit_mol: The molecule input file
        :return: descriptors dictionary
        """

        # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        return {
            "Heavy atoms": Descriptors.HeavyAtomCount(rdkit_mol),
            "H-bond donors": Descriptors.NumHDonors(rdkit_mol),
            "H-bond acceptors": Descriptors.NumHAcceptors(rdkit_mol),
            "Molecular weight": Descriptors.MolWt(rdkit_mol),
            "LogP": Descriptors.MolLogP(rdkit_mol),
        }

    @staticmethod
    def get_smiles(
        rdkit_mol: Chem.Mol,
        isomeric: bool = True,
        explicit_hydrogens: bool = True,
        mapped: bool = False,
    ) -> str:
        """
        Use RDKit to generate a smiles string for the molecule.

        We work with a copy of the input molecule as we may assign an atom map number which will effect the CIP algorithm
        and could break symmetry groups.

        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
            isomeric:
                If the smiles should encode the stereochemistry `True` or not `False`.
            explicit_hydrogens:
                If the smiles should explicitly encode hydrogens `True` or not `False`.
            mapped:
                If the smiles should be mapped to preserve the ordering of the molecule `True` or not `False`.

        Returns:
            A string which encodes the molecule smiles corresponding the the input options.
        """
        cp_mol = copy.deepcopy(rdkit_mol)
        if mapped:
            explicit_hydrogens = True
            for atom in cp_mol.GetAtoms():
                # mapping starts from 1 as 0 means no mapping in rdkit
                atom.SetAtomMapNum(atom.GetIdx() + 1)
        if not explicit_hydrogens:
            cp_mol = Chem.RemoveHs(cp_mol)
        return Chem.MolToSmiles(
            cp_mol, isomericSmiles=isomeric, allHsExplicit=explicit_hydrogens
        )

    @staticmethod
    def get_smirks_matches(rdkit_mol: Chem.Mol, smirks: str) -> List[Tuple[int, ...]]:
        """
        Query the molecule for the tagged smarts pattern (OpenFF SMIRKS).

        Args:
            rdkit_mol:
                The rdkit molecule instance that should be checked against the smarts pattern.
            smirks:
                The tagged SMARTS pattern that should be checked against the molecule.

        Returns:
            A list of atom index tuples which match the corresponding tagged atoms in the smarts pattern.
            Note only tagged atoms indices are returned.
        """
        cp_mol = copy.deepcopy(rdkit_mol)
        smarts_mol = Chem.MolFromSmarts(smirks)
        if smarts_mol is None:
            raise SmartsError(
                f"RDKit could not understand the query {smirks} please check again."
            )
        # we need a mapping between atom map and index in the smarts mol
        # to work out the index of the matched atom
        mapping = {}
        for atom in smarts_mol.GetAtoms():
            smart_index = atom.GetAtomMapNum()
            if smart_index != 0:
                # atom was tagged in the smirks
                mapping[smart_index - 1] = atom.GetIdx()
        # smarts can match forward and backwards so condense the matches
        all_matches = set()
        for match in cp_mol.GetSubstructMatches(
            smarts_mol, uniquify=True, useChirality=True
        ):
            smirks_atoms = [match[atom] for atom in mapping.values()]
            all_matches.add(tuple(smirks_atoms))
        return list(all_matches)

    @staticmethod
    def get_smarts(rdkit_mol: Chem.Mol) -> str:
        """
        Use RDKit to get the smarts string of the molecule.
        :param filename: The molecule input file
        :return: The smarts string
        """

        return Chem.MolToSmarts(rdkit_mol)

    @staticmethod
    def generate_conformers(rdkit_mol: Chem.Mol, conformer_no=10) -> List[np.ndarray]:
        """
        Generate a set of x conformers of the molecule
        :param conformer_no: The amount of conformers made for the molecule
        :param rdkit_mol: The name of the input file
        :return: A list of conformer position arrays
        """

        AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=conformer_no)
        positions = rdkit_mol.GetConformers()

        return [conformer.GetPositions() for conformer in positions]

    @staticmethod
    def find_symmetry_classes(rdkit_mol: Chem.Mol) -> Dict[int, str]:
        """
        Generate list of tuples of symmetry-equivalent (homotopic) atoms in the molecular graph
        based on: https://sourceforge.net/p/rdkit/mailman/message/27897393/
        Our thanks to Dr Michal Krompiec for the symmetrisation method and its implementation.
        :param rdkit_mol: molecule to find symmetry classes for (rdkit mol class object)
        :return: A dict where the keys are the atom indices and the values are their type
        (type is arbitrarily based on index; only consistency is needed, no specific values)
        """

        # Check CIPRank is present for first atom (can assume it is present for all afterwards)
        if not rdkit_mol.GetAtomWithIdx(0).HasProp("_CIPRank"):
            Chem.AssignStereochemistry(
                rdkit_mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
            )

        # Array of ranks showing matching atoms
        cip_ranks = np.array(
            [int(atom.GetProp("_CIPRank")) for atom in rdkit_mol.GetAtoms()]
        )

        # Map the ranks to the atoms to produce a list of symmetrical atoms
        atom_symmetry_classes = [
            np.where(cip_ranks == rank)[0].tolist()
            for rank in range(max(cip_ranks) + 1)
        ]

        # Convert from list of classes to dict where each key is an atom and each value is its class (just a str)
        atom_symmetry_classes_dict = {}
        # i will be used to define the class (just index based)
        for i, sym_class in enumerate(atom_symmetry_classes):
            for atom in sym_class:
                atom_symmetry_classes_dict[atom] = str(i)

        return atom_symmetry_classes_dict

    @staticmethod
    def get_conformer_rmsd(
        rdkit_mol: Chem.Mol, ref_index: int, align_index: int
    ) -> float:
        """
        Get the rmsd between the current rdkit molecule and the coordinates provided
        :param rdkit_mol: rdkit representation of the molecule, conformer 0 is the base
        :param ref_index: the conformer index of the refernce
        :param align_index: the conformer index which should be aligned
        :return: the rmsd value
        """

        return Chem.AllChem.GetConformerRMS(rdkit_mol, ref_index, align_index)

    @staticmethod
    def add_conformer(
        rdkit_mol: Chem.Mol, conformer_coordinates: np.ndarray
    ) -> Chem.Mol:
        """
        Add a new conformation to the rdkit molecule
        :param conformer_coordinates:  A numpy array of the coordinates to be added
        :param rdkit_mol: The rdkit molecule instance
        :return: The rdkit molecule with the conformer added
        """

        conformer = Chem.Conformer()
        for i, coord in enumerate(conformer_coordinates):
            atom_position = Point3D(*coord)
            conformer.SetAtomPosition(i, atom_position)

        rdkit_mol.AddConformer(conformer, assignId=True)

        return rdkit_mol
