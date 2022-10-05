import copy
import re
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Geometry.rdGeometry import Point3D

from qubekit.molecules.components import Atom, Bond, Element
from qubekit.utils.constants import BOHR_TO_ANGS
from qubekit.utils.exceptions import FileTypeError, SmartsError


class RDKit:
    """Class for controlling useful RDKit functions."""

    @staticmethod
    def mol_to_file(rdkit_mol: Chem.Mol, file_name: str) -> None:
        """
        Write the rdkit molecule to the requested file type.
        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
            file_name:
                Name of the file to be created.
        """
        file_path = Path(file_name)
        if file_path.suffix == ".pdb":
            return Chem.MolToPDBFile(rdkit_mol, file_name)
        elif file_path.suffix == ".sdf" or file_path.suffix == ".mol":
            return Chem.MolToMolFile(rdkit_mol, file_name)
        elif file_path.suffix == ".xyz":
            return Chem.MolToXYZFile(rdkit_mol, file_name)
        else:
            raise FileTypeError(
                f"The file type {file_path.suffix} is not supported please chose from xyz, pdb, mol or sdf."
            )

    @staticmethod
    def mol_to_multiconformer_file(rdkit_mol: Chem.Mol, file_name: str) -> None:
        """
        Write the rdkit molecule to a multi conformer file.
        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
            file_name:
                Name of the file to be created.
        """
        file_path = Path(file_name)
        # get the file block writer
        if file_path.suffix == ".pdb":
            writer = Chem.MolToPDBBlock
        elif file_path.suffix == ".mol" or file_path.suffix == ".sdf":
            writer = Chem.MolToMolBlock
        elif file_path.suffix == ".xyz":
            writer = Chem.MolToXYZBlock
        else:
            raise FileTypeError(
                f"The file type {file_path.suffix} is not supported please chose from xyz, pdb, mol or sdf."
            )
        with open(file_name, "w") as out:
            for i in range(rdkit_mol.GetNumConformers()):
                out.write(writer(rdkit_mol, confId=i))

    @staticmethod
    def file_to_rdkit_mol(file_path: Path) -> Chem.Mol:
        """
        Args:
            file_path:
                Path of the file used to generate the rdkit molecule.
        return:
            RDKit molecule object generated from its file (or None if incorrect file type is provided).
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
        Converts smiles strings to RDKit mol object. We are reusing here the OpenFF logic.
        Args:
            smiles_string:
                The hydrogen free smiles string
            name:
                The name of the molecule this will be used when writing the pdb file
        return:
            The RDKit molecule
        """
        mol = AllChem.MolFromSmiles(smiles_string, sanitize=False)
        if name is None:
            name = input("Please enter a name for the molecule:\n>")
        mol.SetProp("_Name", name)

        # strip the atom map before sanitizing and assigning sterochemistry
        atom_index_to_map = {}
        for atom in mol.GetAtoms():
            # set the map back to zero but hide the index in the atom prop data
            atom_index_to_map[atom.GetIdx()] = atom.GetAtomMapNum()
            # set it back to zero
            atom.SetAtomMapNum(0)

        Chem.SanitizeMol(mol)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)

        # Chem.MolFromSmiles adds bond directions (i.e. ENDDOWNRIGHT/ENDUPRIGHT), but
        # doesn't set bond.GetStereo(). We need to call AssignStereochemistry for that.
        Chem.AssignStereochemistry(mol)

        mol = AllChem.AddHs(mol)

        AllChem.EmbedMolecule(mol, randomSeed=1)
        # put the map index back on the atoms
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom_index_to_map.get(atom.GetIdx(), 0))

        return mol

    @staticmethod
    def rdkit_descriptors(rdkit_mol: Chem.Mol) -> Dict[str, float]:
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary.
        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
        returns:
            descriptors dictionary
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

        We work with a copy of the input molecule as we may assign an atom map number which
        will affect the CIP algorithm and could break symmetry groups.

        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
            isomeric:
                If True, the smiles should encode the stereochemistry.
            explicit_hydrogens:
                If True, the smiles should explicitly encode hydrogens.
            mapped:
                If True, the smiles should be mapped to preserve the ordering of the molecule.

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
            smarts_mol, uniquify=False, useChirality=True
        ):
            smirks_atoms = [match[atom] for atom in mapping.values()]
            # add with the lowest index atom first
            if smirks_atoms[0] < smirks_atoms[-1]:
                all_matches.add(tuple(smirks_atoms))
            else:
                all_matches.add(tuple(reversed(smirks_atoms)))
        return list(all_matches)

    @staticmethod
    def get_smarts(rdkit_mol: Chem.Mol) -> str:
        """
        Use RDKit to get the smarts string of the molecule.
        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
        return:
            The smarts string of the molecule
        """

        return Chem.MolToSmarts(rdkit_mol)

    @staticmethod
    def generate_conformers(rdkit_mol: Chem.Mol, conformer_no: int) -> List[np.ndarray]:
        """
        Generate a set of conformers for the molecule including the input conformer.
        Args:
            rdkit_mol:
                A complete Chem.Mol instance of a molecule.
            conformer_no:
                The number of conformers made for the molecule
        return:
            A list of conformer position arrays
        """

        AllChem.EmbedMultipleConfs(
            rdkit_mol,
            numConfs=conformer_no,
            randomSeed=1,
            clearConfs=False,
            useBasicKnowledge=True,
            pruneRmsThresh=1,
            enforceChirality=True,
        )
        positions = rdkit_mol.GetConformers()

        return [conformer.GetPositions() for conformer in positions]

    @staticmethod
    def find_symmetry_classes(rdkit_mol: Chem.Mol) -> Dict[int, str]:
        """
        Generate list of tuples of symmetry-equivalent (homotopic) atoms in the molecular graph
        based on: https://sourceforge.net/p/rdkit/mailman/message/27897393/
        Our thanks to Dr Michal Krompiec for the symmetrisation method and its implementation.
        Args:
            rdkit_mol:
                Molecule to find symmetry classes for (rdkit mol class object)
        return:
            A dict where the keys are the atom indices and the values are their types
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
        Get the rmsd between the current rdkit molecule and the coordinates provided.
        Args:
            rdkit_mol:
                rdkit representation of the molecule, conformer 0 is the base
            ref_index:
                The conformer index of the refernce
            align_index:
                the conformer index which should be aligned
        return:
            The RMSD value
        """

        return Chem.AllChem.GetConformerRMS(rdkit_mol, ref_index, align_index)

    @staticmethod
    def add_conformer(
        rdkit_mol: Chem.Mol, conformer_coordinates: np.ndarray
    ) -> Chem.Mol:
        """
        Add a new conformation to the rdkit molecule.
        Args:
            rdkit_mol:
                The rdkit molecule instance
            conformer_coordinates:
                A numpy array of the coordinates to be added
        return:
            The rdkit molecule with the conformer added
        """

        conformer = Chem.Conformer()
        for i, coord in enumerate(conformer_coordinates):
            atom_position = Point3D(*coord)
            conformer.SetAtomPosition(i, atom_position)

        rdkit_mol.AddConformer(conformer, assignId=True)

        return rdkit_mol


class ReadInput:
    """
    Called inside Ligand; used to handle reading any kind of input valid in QUBEKit
        QC JSON object
        SMILES string
        PDB, MOL2, XYZ file
    """

    def __init__(
        self,
        coords: Optional[np.ndarray] = None,
        rdkit_mol: Optional = None,
        name: Optional[str] = None,
    ):

        self.coords = coords
        self.rdkit_mol = rdkit_mol
        self.name = name

    @classmethod
    def from_smiles(cls, smiles: str, name: Optional[str] = None) -> "ReadInput":
        """
        Make a ReadInput object which can be taken by the Ligand class to make the model.

        Note
        ----
        This method will generate a conformer for the molecule.

        Parameters
        ----------
        smiles:
            The smiles string which should be parsed by rdkit.
        name:
            The name that should be given to the molecule.
        """
        # Smiles string input
        rdkit_mol = RDKit.smiles_to_rdkit_mol(smiles_string=smiles, name=name)
        return cls(name=name, coords=None, rdkit_mol=rdkit_mol)

    @classmethod
    def from_file(cls, file_name: str) -> "ReadInput":
        """
        Read the input file using RDKit and return the molecule data.
        """
        input_file = Path(file_name)
        # if the file is not there raise an error
        if not input_file.exists():
            raise FileNotFoundError(
                f"{input_file.as_posix()} could not be found is this path correct?"
            )
        # xyz is a special case of file only internal readers catch
        if input_file.suffix == ".xyz":
            return cls.from_xyz(file_name=input_file.as_posix())
        # read the input with rdkit
        rdkit_mol = RDKit.file_to_rdkit_mol(file_path=input_file)
        return cls(rdkit_mol=rdkit_mol, coords=None, name=rdkit_mol.GetProp("_Name"))

    @classmethod
    def from_qc_json(cls, qc_json) -> "ReadInput":
        """
        Given a QC JSON object, extracts the topology, atoms and coords of the molecule.
        #TODO we need to be absle to read mapped smiles for this to work with stereochem and aromaticity
        """

        topology = nx.Graph()
        atoms = []

        for i, atom in enumerate(qc_json.symbols):
            atoms.append(
                Atom(
                    atomic_number=Element().number(atom),
                    atom_index=i,
                    atom_name=f"{atom}{i}",
                )
            )
            topology.add_node(i)

        for bond in qc_json.connectivity:
            topology.add_edge(*bond[:2])

        coords = np.array(qc_json.geometry).reshape((len(atoms), 3)) * BOHR_TO_ANGS
        return cls(name=None, rdkit_mol=None, coords=coords)

    @classmethod
    def from_xyz(cls, file_name: str) -> "ReadInput":
        """
        Internal xyz reader.
        Extracts the coords of the molecule.
        """

        traj_molecules = []
        coords = []

        with open(file_name) as xyz_file:
            lines = xyz_file.readlines()

            n_atoms = float(lines[0])

            for line in lines:
                line = line.split()
                # skip frame heading lines
                if len(line) <= 1 or "Iteration" in line:
                    continue

                coords.append([float(line[1]), float(line[2]), float(line[3])])

                if len(coords) == n_atoms:
                    # we have collected the molecule now store the frame
                    traj_molecules.append(np.array(coords))
                    coords = []

        coords = traj_molecules[0] if len(traj_molecules) == 1 else traj_molecules
        return cls(coords=coords, name=None, rdkit_mol=None)


class ReadInputProtein:
    """
    A class that specialises in reading Protein input files.
    #TODO are we better doing this with openmm or another tool?
    """

    def __init__(
        self,
        atoms: List[Atom],
        bonds: Optional[List[Bond]] = None,
        coords: Optional[np.ndarray] = None,
        residues: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        self.atoms = atoms
        self.bonds = bonds
        self.coords = coords
        self.name = name
        self.residues = residues

    @classmethod
    def from_pdb(cls, file_name: str, name: Optional[str] = None):
        """
        Read the protein input pdb file.
        :return:
        """
        with open(file_name, "r") as pdb:
            lines = pdb.readlines()

        coords = []
        atoms = []
        bonds = []
        Residues = []

        # atom counter used for graph node generation
        atom_count = 0
        for line in lines:
            if "ATOM" in line or "HETATM" in line:
                atomic_symbol = str(line[76:78])
                atomic_symbol = re.sub("[0-9]+", "", atomic_symbol).strip()

                # If the element column is missing from the pdb, extract the atomic_symbol from the atom name.
                if not atomic_symbol:
                    atomic_symbol = str(line.split()[2])
                    atomic_symbol = re.sub("[0-9]+", "", atomic_symbol)

                # now make sure we have a valid element
                if atomic_symbol.lower() != "cl" and atomic_symbol.lower() != "br":
                    atomic_symbol = atomic_symbol[0]

                atom_name = f"{atomic_symbol}{atom_count}"
                # TODO should we use a protein pdb package for this?
                qube_atom = Atom(
                    atomic_number=Element().number(atomic_symbol),
                    atom_index=atom_count,
                    atom_name=atom_name,
                    formal_charge=0,
                    aromatic=False,
                    bonds=[],
                )

                atoms.append(qube_atom)

                # also get the residue order from the pdb file so we can rewrite the file
                Residues.append(str(line.split()[3]))

                atom_count += 1
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )

            elif "CONECT" in line:
                conect_terms = line.split()
                for atom in conect_terms[2:]:
                    if int(atom):
                        bond = Bond(
                            atom1_index=int(conect_terms[1]) - 1,
                            atom2_index=int(atom) - 1,
                            bond_order=1,
                            aromatic=False,
                        )
                        bonds.append(bond)
                        atoms[int(conect_terms[1]) - 1].bonds.append(int(atom) - 1)

        coords = np.array(coords)
        residues = [res for res, group in groupby(Residues)]
        if name is None:
            name = Path(file_name).stem
        return cls(
            atoms=atoms,
            bonds=bonds,
            coords=coords,
            residues=residues,
            name=name,
        )
