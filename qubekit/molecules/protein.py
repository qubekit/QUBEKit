import os
from datetime import datetime
from typing import Optional

from qubekit.molecules.components import Bond
from qubekit.molecules.ligand import DefaultsMixin, Molecule
from qubekit.molecules.utils import ReadInputProtein
from qubekit.utils.exceptions import FileTypeError


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
        self._save_to_protein(input_data)

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

    def symmetrise_from_topology(self) -> None:
        """
        First, if rdkit_mol has been generated, get the bond and angle symmetry dicts.
        These will be used by L-J and the Harmonic Bond/Angle params

        Then, based on the molecule topology, symmetrise the methyl / amine hydrogens.
        If there's a carbon, does it have 3/2 hydrogens? -> symmetrise
        If there's a nitrogen, does it have 2 hydrogens? -> symmetrise
        Also keep a list of the methyl carbons and amine / nitrile nitrogens
        then exclude these bonds from the rotatable torsions list.

        TODO This needs to be more applicable to proteins (e.g. if no rdkit_mol is created).
        """

        methyl_hs, amine_hs, other_hs = [], [], []
        methyl_amine_nitride_cores = []
        topology = self.to_topology()
        for atom in self.atoms:
            if atom.atomic_symbol == "C" or atom.atomic_symbol == "N":

                hs = []
                for bonded in topology.neighbors(atom.atom_index):
                    if len(list(topology.neighbors(bonded))) == 1:
                        # now make sure it is a hydrogen (as halogens could be caught here)
                        if self.atoms[bonded].atomic_symbol == "H":
                            hs.append(bonded)

                if (
                    atom.atomic_symbol == "C" and len(hs) == 2
                ):  # This is part of a carbon hydrogen chain
                    other_hs.append(hs)
                elif atom.atomic_symbol == "C" and len(hs) == 3:
                    methyl_hs.append(hs)
                    methyl_amine_nitride_cores.append(atom.atom_index)
                elif atom.atomic_symbol == "N" and len(hs) == 2:
                    amine_hs.append(hs)
                    methyl_amine_nitride_cores.append(atom.atom_index)

        self.symm_hs = {"methyl": methyl_hs, "amine": amine_hs, "other": other_hs}
        self.methyl_amine_nitride_cores = methyl_amine_nitride_cores

    def _save_to_protein(self, mol_input: ReadInputProtein):
        """
        Public access to private file_handlers.py file.
        Users shouldn't ever need to interface with file_handlers.py directly.
        All parameters will be set from a file (or other input) via this public method.
            * Don't bother updating name, topology or atoms if they are already stored.
            * Do bother updating coords, rdkit_mol, residues, Residues, pdb_names
        """

        if mol_input.name is not None:
            self.name = mol_input.name
        if mol_input.atoms is not None:
            self.atoms = mol_input.atoms
        if mol_input.coords is not None:
            self.coordinates = mol_input.coords
        if mol_input.residues is not None:
            self.residues = mol_input.residues
        if mol_input.pdb_names is not None:
            self.pdb_names = mol_input.pdb_names
        if mol_input.bonds is not None:
            self.bonds = mol_input.bonds

        if not self.bonds:
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
            for i, (coord, atom) in enumerate(zip(self.coordinates, self.atoms)):
                x, y, z = coord
                # May cause issues if protein contains more than 10,000 atoms.
                pdb_file.write(
                    f"HETATM {i+1:>4}{atom.atom_name:>5} QUP     1{x:12.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00         {atom.atomic_symbol.upper():>3}\n"
                )

            # Add the connection terms based on the molecule topology.
            for bond in self.bonds:
                pdb_file.write(
                    f"CONECT{bond.atom1_index + 1:5} {bond.atom2_index + 1:5}\n"
                )

            pdb_file.write("END\n")

    def update(self):
        """
        After the protein has been passed to the parametrisation class we get back the bond info
        use this to update all missing terms.
        """
        if not self.bonds:
            self.bonds = []
            # using the new harmonic bond force dict we can add the bond edges to the topology graph
            for bond in self.BondForce:
                self.bonds.append(
                    Bond(
                        atom1_indx=bond[0],
                        atom2_index=bond[1],
                        bond_order=1,
                        aromatic=False,
                    )
                )

        self.symmetrise_from_topology()
