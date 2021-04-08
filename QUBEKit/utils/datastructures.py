import abc
from typing import TYPE_CHECKING, List, Tuple

from pydantic import BaseModel
from typing_extensions import Literal

if TYPE_CHECKING:
    from QUBEKit.molecules import Ligand


class SchemaBase(BaseModel):
    """A basic pydantic starting class which uses assigment validation."""

    class Config:
        validate_assigment = True


class StageBase(SchemaBase, abc.ABC):

    type: Literal["base"] = "base"

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Check any dependencies to make sure that this stage is available to run."""
        ...

    @abc.abstractmethod
    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """The main function of the stage which should perform some parametrisation and return the complete molecule."""
        ...


@dataclasses.dataclass
class LJData:
    a_i: float
    b_i: float
    r_aim: float


@dataclasses.dataclass
class TorsionScan:
    torsion: Tuple[int, int, int, int]
    scan_range: Tuple[int, int]


@dataclasses.dataclass
class GridPointResult:
    """A simple class to help with the torsiondrive json API.
    Important:
        geometries are in bohr
        energy in hartree
    """

    dihedral_angle: int
    input_geometry: List[float]
    final_geometry: List[float]
    final_energy: float
