import re
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from qcelemental.models import Molecule as QCEMolecule
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule
from rdkit.Geometry.rdGeometry import Point3D

from QUBEKit.molecules.components import Atom, Element
from QUBEKit.utils.constants import BOHR_TO_ANGS
from QUBEKit.utils.exceptions import FileTypeError


class RDKit:
    """Class for controlling useful RDKit functions."""

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
        # apply the mol name
        try:
            mol.GetProp("_Name")
        except KeyError:
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
    def get_smiles(rdkit_mol: Chem.Mol) -> str:
        """
        Use RDKit to load in the pdb file of the molecule and get the smiles code.
        :param rdkit_mol: The rdkit molecule
        :return: The smiles string
        """

        return Chem.MolToSmiles(rdkit_mol, isomericSmiles=True, allHsExplicit=True)

    @staticmethod
    def get_smarts(rdkit_mol: Chem.Mol) -> str:
        """
        Use RDKit to get the smarts string of the molecule.
        :param filename: The molecule input file
        :return: The smarts string
        """

        return Chem.MolToSmarts(rdkit_mol)

    @staticmethod
    def to_file(rdkit_mol: Chem.Mol, file_name: str) -> None:
        """
        For the given rdkit molecule write it to the specified file, the type will be geussed from the suffix.
        """
        file_name = Path(file_name)
        if file_name.suffix == ".pdb":
            return Chem.MolToPDBFile(rdkit_mol, file_name)
        elif file_name.suffix == ".mol" or file_name.suffix == ".sdf":
            return Chem.MolToMolFile(rdkit_mol, file_name)

    @staticmethod
    def get_mol(rdkit_mol: Chem.Mol, file_name: str) -> None:
        """
        Use RDKit to generate a mol file.
        :param rdkit_mol: The input rdkit molecule
        :param file_name: The name of the file to write
        :return: The name of the mol file made
        """
        if ".mol" not in file_name:
            mol_name = f"{file_name}.mol"
        else:
            mol_name = file_name
        Chem.MolToMolFile(rdkit_mol, mol_name)

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


class ReadInput:
    """
    Called inside Ligand or Protein; used to handle reading any kind of input valid in QUBEKit
        QC JSON object
        SMILES string
        PDB, MOL2, XYZ file
    :param mol_input: One of the accepted input types:
        QC JSON object
        SMILES string
        PDB, MOL2, XYZ file
    :param name: The name of the molecule. Only necessary for smiles strings but can be
        provided regardless of input type.
    """

    def __init__(
        self,
        topology: Optional[nx.Graph] = None,
        atoms: Optional[List[Atom]] = None,
        coords: Optional[np.ndarray] = None,
        rdkit_mol: Optional = None,
        name: Optional[str] = None,
    ):

        self.topology = topology
        self.atoms = atoms
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
        return cls.from_rdkit(rdkit_mol=rdkit_mol)

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
        return cls.from_rdkit(rdkit_mol=rdkit_mol)

    @classmethod
    def from_rdkit(cls, rdkit_mol, name: Optional[str] = None) -> "ReadInput":
        """
        Using an RDKit Molecule object, extract the name, topology, coordinates and atoms
        """

        if name is None:
            name = rdkit_mol.GetProp("_Name")

        atoms = []
        topology = nx.Graph()
        # Collect the atom names and bonds
        for atom in rdkit_mol.GetAtoms():
            # Collect info about each atom
            atomic_number = atom.GetAtomicNum()
            index = atom.GetIdx()
            try:
                # PDB file extraction
                atom_name = atom.GetMonomerInfo().GetName().strip()
            except AttributeError:
                try:
                    # Mol2 file extraction
                    atom_name = atom.GetProp("_TriposAtomName")
                except KeyError:
                    # smiles and mol files have no atom names so generate them here if they are not declared
                    atom_name = f"{atom.GetSymbol()}{index}"

            qube_atom = Atom(
                atomic_number, index, atom_name, formal_charge=atom.GetFormalCharge()
            )

            # Add the atoms as nodes
            topology.add_node(atom.GetIdx())

            # Add the bonds
            for bonded in atom.GetNeighbors():
                topology.add_edge(atom.GetIdx(), bonded.GetIdx())
                qube_atom.add_bond(bonded.GetIdx())

            # Now add the atom to the molecule
            atoms.append(qube_atom)

        coords = rdkit_mol.GetConformer().GetPositions()
        atoms = atoms or None
        return cls(
            topology=topology,
            atoms=atoms,
            coords=coords,
            rdkit_mol=rdkit_mol,
            name=name,
        )

    @classmethod
    def from_qc_json(cls, qc_json: QCEMolecule) -> "ReadInput":
        """
        Given a QC JSON object, extracts the topology, atoms and coords of the molecule.
        """

        topology = nx.Graph()
        atoms = []

        for i, atom in enumerate(qc_json.atomic_numbers):
            atoms.append(
                Atom(
                    atomic_number=atom,
                    atom_index=i,
                    atom_name=f"{atom}{i}",
                )
            )
            topology.add_node(i)

        for bond in qc_json.connectivity:
            topology.add_edge(*bond[:2])

        coords = np.array(qc_json.geometry).reshape((len(atoms), 3)) * BOHR_TO_ANGS
        atoms = atoms or None
        return cls(topology=topology, atoms=atoms, coords=coords)

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
        return cls(coords=coords, topology=None, atoms=None, rdkit_mol=None)


class ReadInputProtein:
    """
    A class that specialises in reading Protein input files.
    #TODO are we better doing this with openmm or another tool?
    """

    def __init__(
        self,
        topology: Optional[nx.Graph] = None,
        atoms: Optional[List[Atom]] = None,
        coords: Optional[np.ndarray] = None,
        pdb_names: Optional[List[str]] = None,
        residues: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        self.topology = topology
        self.atoms = atoms
        self.coords = coords
        self.name = name
        self.residues = residues
        self.pdb_names = pdb_names

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
        topology = nx.Graph()
        Residues = []
        pdb_names = []

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
                qube_atom = Atom(Element().number(atomic_symbol), atom_count, atom_name)

                atoms.append(qube_atom)

                pdb_names.append(str(line.split()[2]))

                # also get the residue order from the pdb file so we can rewrite the file
                Residues.append(str(line.split()[3]))

                # Also add the atom number as the node in the graph
                topology.add_node(atom_count)
                atom_count += 1
                coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )

            elif "CONECT" in line:
                conect_terms = line.split()
                for atom in conect_terms[2:]:
                    if int(atom):
                        topology.add_edge(int(conect_terms[1]) - 1, int(atom) - 1)

        coords = np.array(coords)
        residues = [res for res, group in groupby(Residues)]
        if name is None:
            name = Path(file_name).stem
        return cls(
            topology=topology,
            atoms=atoms,
            coords=coords,
            pdb_names=pdb_names,
            residues=residues,
            name=name,
        )
