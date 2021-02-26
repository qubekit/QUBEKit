import os
from datetime import datetime
from typing import Optional

import networkx as nx

from QUBEKit.molecules.ligand import DefaultsMixin, Molecule
from QUBEKit.molecules.utils import ReadInputProtein
from QUBEKit.utils.exceptions import FileTypeError


class Protein(DefaultsMixin, Molecule):
    """
    This class handles the protein input to make the QUBEKit xml files and rewrite the pdb so we can use it.
    """

    def __init__(self, mol_input, name=None):
        """
        is_protein      Bool; True for Protein class
        home            Current working directory (location for QUBEKit execution).
        residues        List of all residues in the molecule in order e.g. ['ARG', 'HIS', ... ]
        Residues        List of residue names for each atom e.g. ['ARG', 'ARG', 'ARG', ... 'HIS', 'HIS', ... ]
        pdb_names       List
        """

        super().__init__(mol_input, name)

        self.is_protein = True
        self.home = os.getcwd()
        self.residues = None
        self.Residues = None
        self.pdb_names = None

        self.combination = "opls"

        if not isinstance(mol_input, ReadInputProtein):
            self._check_file_type(file_name=mol_input)
            input_data = ReadInputProtein.from_pdb(file_name=mol_input)
        else:
            input_data = mol_input
        self._save_to_protein(input_data, input_type="input")

    @classmethod
    def from_file(cls, file_name: str, name: Optional[str] = None) -> "Protein":
        """
        Instance the protein class from a pdb file.
        """
        cls._check_file_type(file_name=file_name)
        input_data = ReadInputProtein.from_pdb(file_name=file_name, name=name)
        return cls(input_data)

    @staticmethod
    def _check_file_type(file_name: str) -> None:
        """
        Make sure the protien is being read from a pdb file.
        """
        if ".pdb" not in file_name:
            raise FileTypeError("Proteins can only be read from pdb.")

    def _save_to_protein(self, mol_input: ReadInputProtein, input_type="input"):
        """
        Public access to private file_handlers.py file.
        Users shouldn't ever need to interface with file_handlers.py directly.
        All parameters will be set from a file (or other input) via this public method.
            * Don't bother updating name, topology or atoms if they are already stored.
            * Do bother updating coords, rdkit_mol, residues, Residues, pdb_names
        """

        if mol_input.name is not None:
            self.name = mol_input.name
        if mol_input.topology is not None:
            self.topology = mol_input.topology
        if mol_input.atoms is not None:
            self.atoms = mol_input.atoms
        if mol_input.coords is not None:
            self.coords[input_type] = mol_input.coords
        if mol_input.residues is not None:
            self.residues = mol_input.residues
        if mol_input.pdb_names is not None:
            self.pdb_names = mol_input.pdb_names

        if not self.topology.edges:
            print(
                "No connections found in pdb file; topology will be inferred by OpenMM."
            )
            return
        self.symmetrise_from_topology()

    def write_pdb(self, name=None):
        """
        This method replaces the ligand method as all of the atom names and residue names have to be replaced.
        """

        with open(f"{name if name is not None else self.name}.pdb", "w+") as pdb_file:

            pdb_file.write(f"REMARK   1 CREATED WITH QUBEKit {datetime.now()}\n")
            # Write out the atomic xyz coordinates
            for i, (coord, atom) in enumerate(zip(self.coords["input"], self.atoms)):
                x, y, z = coord
                # May cause issues if protein contains more than 10,000 atoms.
                pdb_file.write(
                    f"HETATM {i+1:>4}{atom.atom_name:>5} QUP     1{x:12.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00         {atom.atomic_symbol.upper():>3}\n"
                )

            # Add the connection terms based on the molecule topology.
            for node in self.topology.nodes:
                bonded = sorted(list(nx.neighbors(self.topology, node)))
                if len(bonded) >= 1:
                    pdb_file.write(
                        f'CONECT{node + 1:5}{"".join(f"{x + 1:5}" for x in bonded)}\n'
                    )

            pdb_file.write("END\n")

    def update(self, input_type="input"):
        """
        After the protein has been passed to the parametrisation class we get back the bond info
        use this to update all missing terms.
        """

        # using the new harmonic bond force dict we can add the bond edges to the topology graph
        for bond in self.HarmonicBondForce:
            self.topology.add_edge(*bond)

        # self.find_angles()
        # self.find_dihedrals()
        # self.find_rotatable_dihedrals()
        # self.find_impropers()
        # self.measure_dihedrals(input_type)
        # self.bond_lengths(input_type)
        # self.measure_angles(input_type)
        # This creates the dictionary of terms that should be symmetrised.
        self.symmetrise_from_topology()
