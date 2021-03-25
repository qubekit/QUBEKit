from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, PositiveInt
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable, PeriodicTable


class Element:
    """
    Simple wrapper class for getting element info using RDKit.
    """

    @staticmethod
    def p_table() -> PeriodicTable:
        return GetPeriodicTable()

    @staticmethod
    def mass(identifier):
        pt = Element.p_table()
        return pt.GetAtomicWeight(identifier)

    @staticmethod
    def number(identifier):
        pt = Element.p_table()
        return pt.GetAtomicNumber(identifier)

    @staticmethod
    def name(identifier):
        pt = Element.p_table()
        return pt.GetElementSymbol(identifier)


class AtomStereoChemistry(str, Enum):
    """
    Atom stereochemistry types.
    """

    R = "R"
    S = "S"
    U = "Unknown"


class BondStereoChemistry(str, Enum):
    """
    Bond stereochemistry types.
    """

    E = "E"
    Z = "Z"
    U = "Unknown"


class Atom(BaseModel):
    """
    Class to hold all of the "per atom" information.
    All atoms in Molecule will have an instance of this Atom class to describe their properties.
    """

    class Config:
        validate_assignment = True
        json_encoders = {Enum: lambda v: v.value}

    atomic_number: PositiveInt = Field(
        ...,
        description="The atomic number of the atom all other properties are based on this number.",
    )
    atom_index: int = Field(
        ..., description="The index this atom has in the molecule object", gt=-1
    )
    atom_name: Optional[str] = Field(
        None,
        description="An optional unqiue atom name that should be assigned to the atom, the ligand object will make sure all atoms have unique names.",
    )
    formal_charge: int = Field(
        ...,
        description="The formal charge of the atom, used to calculate the molecule total charge",
    )
    aromatic: bool = Field(
        ...,
        description="If the atom should be considered aromatic `True` or not `False`.",
    )
    stereochemistry: Optional[AtomStereoChemistry] = Field(
        None,
        description="The stereochemistry of the atom where None means not stereogenic and U is unknown or ambiguous.",
    )
    bonds: Optional[List[int]] = Field(
        None, description="The list of atom indices which are bonded to this atom."
    )

    @classmethod
    def from_rdkit(cls, rd_atom: Chem.Atom) -> "Atom":
        """
        Build a QUBEKit atom from an rdkit atom instance.
        """
        atomic_number = rd_atom.GetAtomicNum()
        index = rd_atom.GetIdx()
        formal_charge = rd_atom.GetFormalCharge()
        aromatic = rd_atom.GetIsAromatic()
        bonds = [a.GetIdx() for a in rd_atom.GetNeighbors()]
        # check for names in the normal places pdb, mol2 and mol
        if rd_atom.HasProp("_Name"):
            name = rd_atom.GetProp("_Name")
        elif rd_atom.HasProp("_TriposAtomName"):
            name = rd_atom.GetProp("_TriposAtomName")
        else:
            try:
                name = rd_atom.GetMonomerInfo().GetName().strip()
            except AttributeError:
                name = None
        # stereochem
        if rd_atom.HasProp("_CIPCode"):
            stereo_code = rd_atom.GetProp("_CIPCode")
        else:
            stereo_code = None
        return cls(
            atomic_number=atomic_number,
            atom_index=index,
            atom_name=name,
            formal_charge=formal_charge,
            aromatic=aromatic,
            stereochemistry=stereo_code,
            bonds=bonds,
        )

    @property
    def atomic_mass(self) -> float:
        """Convert the atomic number to mass."""
        return Element.mass(self.atomic_number)

    @property
    def atomic_symbol(self) -> str:
        """Convert the atomic number to the atomic symbol as per the periodic table."""
        return Element.name(self.atomic_number).title()

    def to_rdkit(self) -> Chem.Atom:
        """
        Convert the QUBEKit atom an RDKit atom.
        """
        # build the atom from atomic number
        rd_atom = Chem.Atom(self.atomic_number)
        rd_atom.SetFormalCharge(self.formal_charge)
        rd_atom.SetIsAromatic(self.aromatic)
        rd_atom.SetProp("_Name", self.atom_name)
        # left is counter clockwise
        if self.stereochemistry == "S":
            rd_atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
        # right is clockwise
        elif self.stereochemistry == "R":
            rd_atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)

        return rd_atom

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.__dict__!r})"
    #
    # def __str__(self):
    #     """
    #     Prints the Atom class objects' names and values one after another with new lines between each.
    #     """
    #
    #     return_str = ""
    #     for key, val in self.__dict__.items():
    #         # Return all objects as {atom object name} = {atom object value(s)}.
    #         return_str += f"\n{key} = {val}\n"
    #
    #     return return_str


class Bond(BaseModel):
    """
    A basic bond class.
    """

    class Config:
        validate_assignment = True
        json_encoders = {Enum: lambda v: v.value}

    atom1_index: int = Field(
        ..., description="The index of the first atom in the bond."
    )
    atom2_index: int = Field(
        ..., description="The index of the second atom in the bond."
    )
    bond_order: float = Field(..., description="The float value of the bond order.")
    aromatic: bool = Field(
        ..., description="If the bond should be considered aromatic."
    )
    stereochemistry: Optional[BondStereoChemistry] = Field(
        None,
        description="The stereochemistry of the bond, where None means not stereogenic.",
    )

    @classmethod
    def from_rdkit(cls, rd_bond: Chem.Bond) -> "Bond":
        """
        Build a QUBEKit bond class from an rdkit reference.
        """
        atom1_index = rd_bond.GetBeginAtomIdx()
        atom2_index = rd_bond.GetEndAtomIdx()
        aromatic = rd_bond.GetIsAromatic()
        order = rd_bond.GetBondTypeAsDouble()
        stereo_tag = rd_bond.GetStereo()
        if stereo_tag == Chem.BondStereo.STEREOZ:
            stereo = "Z"
        elif stereo_tag == Chem.BondStereo.STEREOE:
            stereo = "E"
        else:
            stereo = None
        return cls(
            atom1_index=atom1_index,
            atom2_index=atom2_index,
            aromatic=aromatic,
            bond_order=order,
            stereochemistry=stereo,
        )

    @property
    def rdkit_type(self) -> Chem.BondType:
        """
        Convert the bond order float to a bond type.
        """
        conversion = {
            1: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.QUADRUPLE,
            5: Chem.BondType.QUINTUPLE,
            6: Chem.BondType.HEXTUPLE,
            7: Chem.BondType.ONEANDAHALF,
        }
        return conversion[self.bond_order]

    @property
    def rdkit_stereo(self) -> Optional[Chem.BondStereo]:
        """
        Return the rdkit style stereo enum.
        """
        if self.stereochemistry == "E":
            return Chem.BondStereo.STEREOE
        elif self.stereochemistry == "Z":
            return Chem.BondStereo.STEREOZ
        return None
