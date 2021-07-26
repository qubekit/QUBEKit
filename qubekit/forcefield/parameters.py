"""
Build up a list of forcefield objects to be stored in the main forcefield model
"""
import abc
import decimal
from typing import Dict, Set, Tuple

from pydantic import BaseModel, Field, validator
from typing_extensions import Literal

from qubekit.utils.constants import PI

Point3 = Tuple[float, float, float]
Point4 = Tuple[float, float, float, float]


class BasicParameterModel(BaseModel, abc.ABC):

    type: Literal["base"] = "base"
    attributes: Set[str] = Field(
        set(),
        description="Any cosmetic attributes which should be added to the xml such as tagging parameters for fitting with ForceBalance.",
    )

    class Config:
        validate_assignment = True
        extra = "forbid"

    @abc.abstractmethod
    def xml_data(self) -> Dict[str, str]:
        """Format the data for openmm xml."""
        ...

    @classmethod
    @abc.abstractmethod
    def openmm_type(cls) -> str:
        """The string tag that should be used in the xml format."""
        ...

    def update(self, **kwargs):
        """Update the parameter attributes which are valid for this parameter."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BaseParameter(BasicParameterModel):

    atoms: Tuple[int, ...] = Field(
        ..., description="The atom indices that this potential applies to."
    )

    @validator("atoms")
    def _check_atoms(cls, atoms: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(atoms) != cls._n_tags():
            raise ValueError(
                "The number of target atoms for this potential is incorrect."
            )
        return atoms

    @classmethod
    @abc.abstractmethod
    def _n_tags(cls) -> int:
        """The number of atoms which should be tagged by this parameter."""
        ...

    def xml_data(self) -> Dict[str, str]:
        """Get the xml data for this parameter in the correct format."""
        data = self.dict(exclude={"atoms", "attributes", "type"})
        # we need to string everything
        for key, value in data.items():
            if value == "pi":
                new_value = str(PI)
            else:
                new_value = str(value)
            data[key] = new_value
        # do fitting extras for force balance
        attribute_string = ", ".join(
            attribute for attribute in self.attributes if self.attributes
        )
        if attribute_string:
            data["parameterize"] = attribute_string
        # now do class tags
        for i, atom in enumerate(self.atoms, start=1):
            data[f"class{i}"] = str(atom)
        return data


class HarmonicBondParameter(BaseParameter):

    type: Literal["HarmonicBondParameter"] = "HarmonicBondParameter"
    length: float = Field(
        ...,
        description="The equilibrium bond length for a harmonic bond potential in nanometers.",
    )
    k: float = Field(
        ...,
        description="The force constant in a harmonic bond potential in kj/mol/ nm.",
    )

    @classmethod
    def _n_tags(cls) -> int:
        return 2

    @classmethod
    def openmm_type(cls) -> str:
        return "Bond"


class HarmonicAngleParameter(BaseParameter):

    type: Literal["HarmonicAngleParameter"] = "HarmonicAngleParameter"
    k: float = Field(
        ...,
        description="The force constant for a harmonic angle potential in kj/mol/ rad**2.",
    )
    angle: float = Field(
        ..., description="The equilibrium angle in a harmonic potential in rad."
    )

    @classmethod
    def _n_tags(cls) -> int:
        return 3

    @classmethod
    def openmm_type(cls) -> str:
        return "Angle"


class PeriodicTorsionParameter(BaseParameter):

    type: Literal["PeriodicTorsionParameter"] = "PeriodicTorsionParameter"
    k1: float = 0
    k2: float = 0
    k3: float = 0
    k4: float = 0
    periodicity1: int = 1
    periodicity2: int = 2
    periodicity3: int = 3
    periodicity4: int = 4
    phase1: float = 0
    phase2: float = PI
    phase3: float = 0
    phase4: float = PI

    @classmethod
    def _n_tags(cls) -> int:
        return 4

    @classmethod
    def openmm_type(cls) -> str:
        return "Proper"


class ImproperTorsionParameter(PeriodicTorsionParameter):

    type: Literal["ImproperTorsionParameter"] = "ImproperTorsionParameter"

    @classmethod
    def openmm_type(cls) -> str:
        return "Improper"


class BasicRBTorsionParameter(BaseParameter):

    type: Literal["BasicRBTorsionParameter"] = "BasicRBTorsionParameter"
    c0: float = 0
    c1: float = 0
    c2: float = 0
    c3: float = 0
    c4: float = 0
    c5: float = 0

    @classmethod
    def _n_tags(cls) -> int:
        return 4


class ProperRBTorsionParameter(BasicRBTorsionParameter):

    type: Literal["ProperRBTorsionParameter"] = "ProperRBTorsionParameter"

    @classmethod
    def openmm_type(cls) -> str:
        return "Proper"


class ImproperRBTorsionParameter(BasicRBTorsionParameter):

    type: Literal["ImproperRBTorsionParameter"] = "ImproperRBTorsionParameter"

    @classmethod
    def openmm_type(cls) -> str:
        return "Improper"


class BasicNonBondedParameter(BaseParameter):

    type: Literal["basic_non_bonded"] = "basic_non_bonded"
    charge: decimal.Decimal = Field(
        ..., description="The atomic partial charge in elementary charge units."
    )

    @classmethod
    def openmm_type(cls) -> str:
        return "Atom"

    @classmethod
    def _n_tags(cls) -> int:
        return 1

    def xml_data(self) -> Dict[str, str]:
        xml_data = super().xml_data()
        # we need to swap class for type
        class_type = xml_data.pop("class1")
        xml_data["type"] = "QUBE_" + class_type
        return xml_data


class LennardJones612Parameter(BasicNonBondedParameter):

    type: Literal["LennardJones612"] = "LennardJones612"
    epsilon: float = Field(
        ...,
        description="The atomic epsilon parameter which defined the well depth in kj/mol.",
    )
    sigma: float = Field(
        ...,
        description="The atomic sigma parameter which defines the minimum energy distance in nm.",
    )


class VirtualSite3Point(BasicParameterModel):

    type: Literal["VirtualSite3Point"] = "VirtualSite3Point"
    parent_index: int = Field(
        ..., description="The index of the atom the site is attached to."
    )
    closest_a_index: int = Field(
        ...,
        description="The index of the next closest atom used to construct the site position.",
    )
    closest_b_index: int = Field(
        ...,
        description="The index of the 2 closest atom which should be used to construct the site possition.",
    )
    o_weights: Point3 = Field(
        ..., description="The weights that should be used to build the origin point."
    )
    x_weights: Point3 = Field(
        ..., description="The weights that should be used to build the x unit vector."
    )
    y_weights: Point3 = Field(
        ..., description="The weights that should be used to build y unit vector."
    )
    p1: float = Field(
        ...,
        description="The position of the site in x defined by the local unit vectors.",
    )
    p2: float = Field(
        ...,
        description="The position of the site in y defined by the local unit vectors.",
    )
    p3: float = Field(
        ...,
        description="The position of the site in z defined by the local unit vectors.",
    )
    charge: decimal.Decimal = Field(..., description="The charge of the site.")

    @classmethod
    def _n_tags(cls) -> int:
        return 1

    @classmethod
    def openmm_type(cls) -> str:
        return "VirtualSite"

    def xml_data(self) -> Dict[str, str]:
        """A special method used to format virtual sites."""
        basic_data = self.dict(include={"p1", "p2", "p3"})
        basic_data.update(
            {
                "atom1": self.parent_index,
                "atom2": self.closest_a_index,
                "atom3": self.closest_b_index,
                "type": "localCoords",
            }
        )
        # now do special list formatting
        for i, weight in enumerate(self.o_weights, start=1):
            basic_data[f"wo{i}"] = weight
        for i, weight in enumerate(self.x_weights, start=1):
            basic_data[f"wx{i}"] = weight
        for i, weight in enumerate(self.y_weights, start=1):
            basic_data[f"wy{i}"] = weight
        # now string everything
        for key, value in basic_data.items():
            basic_data[key] = str(value)
        return basic_data


class VirtualSite4Point(VirtualSite3Point):

    type: Literal["VirtualSite4Point"] = "VirtualSite4Point"
    closest_c_index: int = Field(
        ...,
        description="The index of the third closes atom used to build a 4 point site model.",
    )
    o_weights: Point4
    x_weights: Point4
    y_weights: Point4

    def xml_data(self) -> Dict[str, str]:
        """Extend the special xml method to get the extra reference atom in the xml data."""
        data = super().xml_data()
        # now add the new term
        data["atom4"] = str(self.closest_c_index)
        return data
