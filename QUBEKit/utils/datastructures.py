#!/usr/bin/env python3

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator
from qcelemental.models.types import Array
from rdkit.Chem.rdchem import GetPeriodicTable, PeriodicTable

from QUBEKit.utils.exceptions import MissingReferenceData, TorsionDriveDataError


class CustomNamespace(SimpleNamespace):
    """
    Adds iteration and dict-style access of keys, values and items to SimpleNamespace.
    TODO Add get() method? (similar to dict)
    """

    def keys(self):
        for key in self.__dict__:
            yield key

    def values(self):
        for value in self.__dict__.values():
            yield value

    def items(self):
        for key, value in self.__dict__.items():
            yield key, value

    def __iter__(self):
        return self.items()


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


class Atom:
    """
    Class to hold all of the "per atom" information.
    All atoms in Molecule will have an instance of this Atom class to describe their properties.
    """

    def __init__(
        self,
        atomic_number: int,
        atom_index: int,
        atom_name: str = "",
        partial_charge: Optional[float] = None,
        formal_charge: Optional[int] = None,
        atom_type: Optional[str] = None,
        bonds: Optional[List[int]] = None,
    ):

        self.atomic_number = atomic_number
        # The QUBEKit assigned name derived from the atomic name and its index e.g. C1, F8, etc
        self.atom_name = atom_name
        self.atom_index = atom_index
        self.partial_charge = partial_charge
        self.formal_charge = formal_charge
        self.atom_type = atom_type
        self.bonds = bonds or []

    @property
    def atomic_mass(self) -> float:
        """Convert the atomic number to mass."""
        return Element.mass(self.atomic_number)

    @property
    def atomic_symbol(self) -> str:
        """Convert the atomic number to the atomic symbol as per the periodic table."""
        return Element.name(self.atomic_number).title()

    def add_bond(self, bonded_index: int) -> None:
        """
        Add a bond to the atom, this will make sure the bond has not already been described
        :param bonded_index: The index of the atom bonded to self
        """

        if bonded_index not in self.bonds:
            self.bonds.append(bonded_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self):
        """
        Prints the Atom class objects' names and values one after another with new lines between each.
        """

        return_str = ""
        for key, val in self.__dict__.items():
            # Return all objects as {atom object name} = {atom object value(s)}.
            return_str += f"\n{key} = {val}\n"

        return return_str


class ExtraSite:
    """
    Used to store extra sites for xml writer in ligand.
    This class is used by both internal v-sites fitting and the ONETEP reader.
    """

    def __init__(self):
        self.parent_index: Optional[int] = None
        self.closest_a_index: Optional[int] = None
        self.closest_b_index: Optional[int] = None
        # Optional: Used for Nitrogen only.
        self.closest_c_index: Optional[int] = None

        self.o_weights: Optional[List[float]] = None
        self.x_weights: Optional[List[float]] = None
        self.y_weights: Optional[List[float]] = None

        self.p1: Optional[float] = None
        self.p2: Optional[float] = None
        self.p3: Optional[float] = None
        self.charge: Optional[float] = None


class ReferenceData(BaseModel):
    """
    A basic reference data class.

    Note
    ----
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
        [-165, 180], description="The torsion range this dihedral was scanned over."
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

    @validator("torsion_drive_range")
    def _check_scan_range(cls, scan_range: Tuple[int, int]) -> Tuple[int, int]:
        """
        Make sure the scan range is in order lowest to highest.
        """
        return tuple(sorted(scan_range))

    @property
    def max_angle(self) -> int:
        return self.torsion_drive_range[1]

    @property
    def min_angle(self) -> int:
        return self.torsion_drive_range[0]

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

        Parameters
        ----------
        qdata_file
            The path to the qdata file which should be read, be default this is qdata.txt.
        dihedral
            The indices of the atoms which make up the target dihedral.
        grid_spacing
            The spacing of the dihedral expected in the input file.
        torsion_drive_range
            The torsion angle scan range expected in the input file.


        Returns
        -------
        TorsionDriveData
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
        torsion_data._validate_angles()
        return torsion_data

    @staticmethod
    def _measure_angle(coords: np.ndarray, dihedral: Tuple[int, int, int, int]) -> int:
        """
        For the given set of coords and dihedral atom indices measure the angle.

        Parameters
        ----------
        coords
            The coordinates for which the dihedral should be measured in the shape (n, 3), where n is the number of atoms.
        dihedral
            The indices of the atoms in the target dihedral.

        Note
        ----
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

    def _validate_angles(self):
        """
        Make sure that the grid spacing and torsion scan range are consistent with the stored refernce data.
        """
        for i in range(
            self.min_angle, self.max_angle + self.grid_spacing, self.grid_spacing
        ):
            if i not in self.reference_data:
                raise MissingReferenceData(
                    f"the torsion angle {i} has no reference data but is required for fitting."
                )

    def add_grid_point(self, grid_data: TorsionData) -> None:
        """
        Add some grid point data to the torsion drive dataset. Here we make sure an angle only appears once and is within
        the specified scan range.
        """
        possible_angles = [
            i
            for i in range(
                self.min_angle, self.max_angle + self.grid_spacing, self.grid_spacing
            )
        ]
        if grid_data.angle in possible_angles:
            self.reference_data[grid_data.angle] = grid_data
        else:
            raise TorsionDriveDataError(
                f"Can not add data for torsion angle {grid_data.angle} as it is not consistent with a scan range off {self.torsion_drive_range} and grid spacing {self.grid_spacing}"
            )

    @property
    def central_bond(self) -> Tuple[int, int]:
        """
        Returns
        -------
        tuple
            The central bond tuple of the scanned torsion.
        """
        return tuple(self.dihedral[1:3])

    def create_qdata(self, file_name: str = "qdata.txt") -> None:
        """
        Create a qdata file from the reference data stored in the class.
        """
        with open(file_name, "w") as qdata:
            for i, result in enumerate(self.reference_data.values()):
                qdata.write(f"JOB {i}\n")
                qdata.write(
                    f"COORDS  {'  '.join(str(i) for i in  result.geometry.flatten().tolist())}\n"
                )
                qdata.write(f"ENERGY {result.energy}\n\n")


class ParameterTags(BaseModel):
    """
    A class that helps the forcefield builder tag certain parameters with cosmetic attributes.
    """

    HarmonicBondForce_groups: List[Tuple[int, int]] = Field(
        [], description="The list of bond tuples which should be tagged."
    )
    HarmonicBondForce_tags: Dict[str, str] = Field(
        {},
        description="The tags that should be added to each bond parameter, in the style key=value.",
    )
    HarmonicAngleForce_groups: List[Tuple[int, int, int]] = Field(
        [], description="The list of angle tuples which should be tagged."
    )
    HarmonicAngleForce_tags: Dict[str, str] = Field(
        {}, description="The tags that should be added to each angle parameter."
    )
    PeriodicTorsionForce_groups: List[Tuple[int, int, int, int]] = Field(
        [], description="The list of dihedral tuples which should be tagged."
    )
    PeriodicTorsionForce_tags: Dict[str, str] = Field(
        {}, description="The tags which should added to each torsion parameter."
    )

    @staticmethod
    def _check_key(key: Tuple[int, ...], group: List[Tuple[int, ...]]) -> bool:
        """
        A general key check function that can be used to see if the given key is in the force group list.
        """
        rev_key = tuple(reversed(key))
        if key in group or rev_key in group:
            return True

        return False

    def check_bond_group(self, bond: Tuple[int, int]) -> bool:
        """
        Check to see if the bond tuple should be tagged or not.
        """
        return self._check_key(key=bond, group=self.HarmonicBondForce_groups)

    def check_angle_group(self, angle: Tuple[int, int, int]) -> bool:
        """
        Check to see if the angle tuple should be tagged or not.
        """
        return self._check_key(key=angle, group=self.HarmonicAngleForce_groups)

    def check_torsion_group(self, torsion: Tuple[int, int, int, int]) -> bool:
        """
        Check to see if the torsion should be tagged or not.
        """
        return self._check_key(key=torsion, group=self.PeriodicTorsionForce_groups)
