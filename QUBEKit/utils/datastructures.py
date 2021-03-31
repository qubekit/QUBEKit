import abc

from pydantic import BaseModel
from typing_extensions import Literal

from QUBEKit.molecules import Ligand


class SchemaBase(BaseModel):
    """A basic pydantic starting class which uses assigment validation."""

    class Config:
        validate_assigment = True


class StageBase(BaseModel, abc.ABC):

    type: Literal["base"] = "base"

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Check any dependencies to make sure that this stage is available to run."""
        ...

    @abc.abstractmethod
    def run(self, molecule: Ligand, **kwargs) -> Ligand:
        """The main function of the stage which should perform some parametrisation and return the complete molecule."""
        ...
