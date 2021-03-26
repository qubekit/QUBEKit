import abc
from types import SimpleNamespace

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
