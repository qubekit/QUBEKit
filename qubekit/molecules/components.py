from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, dataclasses, validator
from qcelemental.models.types import Array
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable, PeriodicTable

from qubekit.utils.exceptions import MissingReferenceData, TorsionDriveDataError


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


@dataclasses.dataclass  # Cannot be frozen as params are loaded separately.
class AIM:
    """Data in atomic units."""

    volume: Optional[float] = None
    charge: Optional[float] = None
    c8: Optional[float] = None
    # TODO Extend to include other types of potential e.g. Buckingham


@dataclasses.dataclass(frozen=True)
class Dipole:
    """Data in atomic units."""

    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclasses.dataclass(frozen=True)
class Quadrupole:
    """Data in atomic units."""

    q_xx: float
    q_xy: float
    q_xz: float
    q_yz: float
    q_yy: float
    q_zz: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                [self.q_xx, self.q_xy, self.q_xz],
                [self.q_xy, self.q_yy, self.q_yz],
                [self.q_xz, self.q_yz, self.q_zz],
            ]
        )


@dataclasses.dataclass(frozen=True)
class CloudPen:
    """Data in atomic units."""

    a: float
    b: float


class Atom(BaseModel):
    """
    Class to hold all of the "per atom" information.
    All atoms in Molecule will have an instance of this Atom class to describe their properties.
    """

    class Config:
        validate_assignment = True
        json_encoders = {Enum: lambda v: v.value}

    atomic_number: int = Field(
        ...,
        description="The atomic number of the atom all other properties are based on this number.",
        ge=0,
    )
    atom_index: int = Field(
        ...,
        description="The index this atom has in the molecule object",
        ge=0,
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
    aim: Optional[AIM] = Field(
        AIM(),
    )
    dipole: Optional[Dipole] = Field(
        None,
    )
    quadrupole: Optional[Quadrupole] = Field(
        None,
    )
    cloud_pen: Optional[CloudPen] = Field(
        None,
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

    @property
    def indices(self) -> Tuple[int, int]:
        return self.atom1_index, self.atom2_index


class ReferenceData(BaseModel):
    """
    A basic reference data class.

    Note:
        Mixed units are used here due to the strange units in qdata.txt files.
    """

    geometry: Array[np.ndarray] = Field(
        ..., description="The geometry of the single point result in angstrom."
    )
    energy: float = Field(
        ..., description="The qm calculated energy for this geometry in hartree."
    )
    gradient: Optional[Array[np.ndarray]] = Field(
        None,
        description="The gradient calculated at this geometry in hartree per bohr.",
    )

    class Config:
        validate_assignment = True
        json_encoders = {np.ndarray: lambda v: v.flatten().tolist()}

    @validator("geometry", "gradient")
    def _check_geom_grad(cls, array: np.ndarray) -> np.array:
        """
        Make sure that the geometry or gradient array is in the correct shape.
        """
        if array is None:
            return array
        else:
            return array.reshape((-1, 3))


class TorsionData(ReferenceData):
    """
    This is a reference data class for a grid point in a torsion drive scan.
    This class extends the normal data class with an angle attribute.
    """

    angle: int = Field(..., description="The angle this data point was calculated at.")


class TorsionDriveData(BaseModel):
    """
    A container class to help store torsiondrive reference data the class is locked once made to ensure the data is valid.
    """

    grid_spacing: int = Field(
        15, description="The angle between grid points on a torsion drive."
    )
    torsion_drive_range: Tuple[int, int] = Field(
        (-165, 180), description="The torsion range this dihedral was scanned over."
    )
    dihedral: Tuple[int, int, int, int] = Field(
        ...,
        description="The dihedral which was targeted during the torsion drive for fitting.",
    )
    reference_data: Dict[int, TorsionData] = Field(
        {},
        description="The list of single point torsion data points which detail the geometry, angle and energy.",
    )

    class Config:
        validate_assignment = True
        allow_mutation = False
        json_encoders = {np.ndarray: lambda v: v.flatten().tolist()}

    @validator("torsion_drive_range")
    def _check_scan_range(cls, scan_range: Tuple[int, int]) -> Tuple[int, int]:
        """
        Make sure the scan range is in order lowest to highest.
        """
        return tuple(sorted(scan_range))

    def to_file(self, file_name: str) -> None:
        """
        Write the object to file.
        """
        with open(file_name, "w") as output:
            output.write(self.json(indent=2))

    @property
    def max_angle(self) -> int:
        return self.torsion_drive_range[1]

    @property
    def min_angle(self) -> int:
        return self.torsion_drive_range[0]

    @property
    def possible_angles(self) -> List[int]:
        """
        Get all of the possible angle values for this grid spacing and scan range combination.
        """
        angles = [
            i
            for i in range(
                self.max_angle, self.min_angle - self.grid_spacing, -self.grid_spacing
            )
            if i >= self.min_angle
        ]
        return angles

    @classmethod
    def from_qdata(
        cls,
        dihedral: Tuple[int, int, int, int],
        qdata_file: str = "qdata.txt",
        grid_spacing: int = 15,
        torsion_drive_range: Tuple[int, int] = (-165, 180),
    ) -> "TorsionDriveData":
        """
        Create a TorsionDrive Data class from a qdata.txt file and the target dihedral.

        Args:
            qdata_file: The path to the qdata file which should be read, be default this is qdata.txt.
            dihedral: The indices of the atoms which make up the target dihedral.
            grid_spacing: The spacing of the dihedral expected in the input file.
            torsion_drive_range: The torsion angle scan range expected in the input file.

        Returns:
            An instance of the class which holds the torsion drive data.
        """
        energies = []
        geometries = []
        angles = []
        # get the geometries and energies
        with open(qdata_file, "r") as qdata:
            for line in qdata.readlines():
                if "ENERGY" in line:
                    energies.append(float(line.split()[1]))
                elif "COORDS" in line:
                    coords = [float(x) for x in line.split()[1:]]
                    coords = np.array(coords).reshape((-1, 3))
                    # get the angle
                    angle = cls._measure_angle(coords=coords, dihedral=dihedral)
                    angles.append(angle)
                    geometries.append(coords)

        torsion_data = cls(
            grid_spacing=grid_spacing,
            torsion_drive_range=torsion_drive_range,
            dihedral=dihedral,
        )
        # now for each grid point at the reference data
        for i, angle in enumerate(angles):
            data_point = TorsionData(
                geometry=geometries[i], energy=energies[i], angle=angle
            )
            torsion_data.add_grid_point(grid_data=data_point)

        # make sure all of the reference data is consistent
        torsion_data.validate_angles()
        return torsion_data

    @staticmethod
    def _measure_angle(coords: np.ndarray, dihedral: Tuple[int, int, int, int]) -> int:
        """
        For the given set of coords and dihedral atom indices measure the angle.

        Args:
            coords: The coordinates for which the dihedral should be measured in the shape (n, 3), where n is the number of atoms.
            dihedral: The indices of the atoms in the target dihedral.

        Note:
            The normal expected scan range for a torsiondrive is ~-170 to 180 so we flip a measured angle of -180 to 180.
        """
        # Calculate the dihedral angle in the molecule using the molecule data array.
        x1, x2, x3, x4 = [coords[atom_id] for atom_id in dihedral]
        b1, b2, b3 = x2 - x1, x3 - x2, x4 - x3
        t1 = np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3))
        t2 = np.dot(np.cross(b1, b2), np.cross(b2, b3))
        angle = round(np.degrees(np.arctan2(t1, t2)))
        # work around symmetry at 180
        if angle == -180:
            return 180
        else:
            return angle

    def validate_angles(self):
        """
        Make sure that the grid spacing and torsion scan range are consistent with the stored refernce data.
        """
        for i in self.possible_angles:
            if i not in self.reference_data:
                raise MissingReferenceData(
                    f"the torsion angle {i} has no reference data but is required for fitting."
                )

    def add_grid_point(self, grid_data: TorsionData) -> None:
        """
        Add some grid point data to the torsion drive dataset. Here we make sure an angle only appears once and is within
        the specified scan range.
        """
        if grid_data.angle in self.possible_angles:
            self.reference_data[grid_data.angle] = grid_data
        else:
            raise TorsionDriveDataError(
                f"Can not add data for torsion angle {grid_data.angle} as it is not consistent with a scan range off {self.torsion_drive_range} and grid spacing {self.grid_spacing}"
            )

    @property
    def central_bond(self) -> Tuple[int, int]:
        """
        Returns:
            The central bond tuple of the scanned torsion.
        """
        return tuple(self.dihedral[1:3])
