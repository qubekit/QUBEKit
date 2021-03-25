import abc
from typing import Dict, Generator, List, Optional, Tuple, Type

from pydantic import BaseModel, Field
from typing_extensions import Literal

from QUBEKit.forcefield.parameters import (
    BaseParameter,
    HarmonicAngleParameter,
    HarmonicBondParameter,
    ImproperTorsionParameter,
    LennardJones612Parameter,
    PeriodicTorsionParameter,
)
from QUBEKit.utils.exceptions import MissingParameterError


class BaseForceGroup(BaseModel):
    """
    The base force group all should derive from this is used to query and set parameters.
    """

    type: Literal["base"] = "base"
    parameters: Optional[Dict[Tuple[int, ...], Type[BaseParameter]]] = None

    @classmethod
    @abc.abstractmethod
    def _parameter_class(cls):
        """The parameter schema for this force group."""
        ...

    @classmethod
    @abc.abstractmethod
    def openmm_group(cls):
        """The openMM friendly group name."""
        ...

    @classmethod
    @abc.abstractmethod
    def symmetry_parameters(cls) -> List[str]:
        """
        Get a list of the per parameter attributes which this force group declares are safe to apply symmetry to.
        """
        ...

    @property
    def n_parameters(self) -> int:
        if self.parameters is None:
            return 0
        else:
            return len(self.parameters)

    class Config:
        validate_assignment = True

    @property
    def iter_parameters(self) -> Generator[Type[BaseParameter], None, None]:
        """
        An iterator over the parameters in the force group.
        """
        if self.parameters is None:
            parameters = []
        else:
            parameters = self.parameters.values()
        for parmaeter in parameters:
            yield parmaeter

    def xml_data(self) -> Dict[str, str]:
        """
        Create the required xml data dictionary for this forcegroup which can be converted to OpenMM xml.
        """
        attrib = self.dict(exclude={"parameters", "type"})
        for key, value in attrib.items():
            attrib[key] = str(value)
        return attrib

    def set_parameter(self, atoms: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        """
        Set a parameter in this force group.

        Check to see if we already have a parameter for this interaction, if we do pull it out and update with the new values, else create
        a new parameter with the input values.

        Args:
            atoms: The tuple of atom indices the parameter applies to.
            kwargs: Any parameter specific attributes which should be used to update or construct the parameter object.

        Returns:
            The tuple of atom indices the parameter is registered under note this might be different to that supplied due to ordering.
        """
        if self.parameters is None:
            self.parameters = {}
        try:
            parameter = self.get_parameter(atoms=atoms)
            parameter.update(**kwargs)
        except MissingParameterError:
            parameter_type = self._parameter_class()
            kwargs["atoms"] = atoms
            parameter = parameter_type(**kwargs)

        self.parameters[atoms] = parameter
        return parameter.atoms

    def clear_parameters(self) -> None:
        """Remove all current parameters."""
        self.parameters = {}

    def remove_parameter(self, atoms: Tuple[int, ...]) -> Type[BaseParameter]:
        """Remove a parameter in this force group."""
        parameter = self.get_parameter(atoms)
        parameter = self.parameters.pop(parameter.atoms)
        return parameter

    def get_parameter(self, atoms: Tuple[int, ...]) -> Type[BaseParameter]:
        """
        Find a parameter in this force group and return it.

        Args:
            atoms: The tuple of atom indices that we want to try and find a parameter for.

        Important:
            This is not a copy of the parameter so any changes will update the forcefield.

        Returns:
            A parameter of type self._parameter_class for the requested atoms.

        Raises:
            MissingParameter: When there is no parameter covering the requested atoms.
        """
        try:
            parameter = self.parameters[atoms]
        except KeyError:
            try:
                parameter = self.parameters[tuple(reversed(atoms))]
            except KeyError:
                raise MissingParameterError(
                    f"No parameter was found for a potential between atoms {atoms}."
                )
        return parameter


class HarmonicBondForce(BaseForceGroup):

    type: Literal["HarmonicBondForce"] = "HarmonicBondForce"

    @classmethod
    def _parameter_class(cls):
        return HarmonicBondParameter

    @classmethod
    def openmm_group(cls):
        return "HarmonicBondForce"

    @classmethod
    def symmetry_parameters(cls) -> List[str]:
        return ["k", "length"]


class HarmonicAngleForce(BaseForceGroup):

    type: Literal["HarmonicAngleForce"] = "HarmonicAngleForce"

    @classmethod
    def _parameter_class(cls):
        return HarmonicAngleParameter

    @classmethod
    def openmm_group(cls):
        return "HarmonicAngleForce"

    @classmethod
    def symmetry_parameters(cls) -> List[str]:
        return ["angle", "k"]


class PeriodicTorsionForce(BaseForceGroup):

    type: Literal["PeriodicTorsionForce"] = "PeriodicTorsionForce"
    ordering: Literal["default", "amber", "charmm", "smirnoff"] = "default"

    @classmethod
    def _parameter_class(cls):
        return PeriodicTorsionParameter

    @classmethod
    def openmm_group(cls):
        return "PeriodicTorsionForce"

    @classmethod
    def symmetry_parameters(cls) -> List[str]:
        """Not sure we should change anything here so keep at default."""
        return []


class ImproperTorsionForce(BaseForceGroup):

    type: Literal["ImproperTorsionForce"] = "ImproperTorsionForce"

    @classmethod
    def _parameter_class(cls):
        return ImproperTorsionParameter

    @classmethod
    def openmm_group(cls):
        return "PeriodicTorsionForce"

    @classmethod
    def symmetry_parameters(cls) -> List[str]:
        return []


class LennardJones126Force(BaseForceGroup):

    type: Literal["NonbondedForce"] = "NonbondedForce"
    coulomb14scale: float = Field(0.8333333333, description="The 1-4 coulomb scaling.")
    lj14scale: float = Field(0.5, description="The 1-4 lj scaling.")
    combination: Literal["amber"] = Field(
        "amber", description="The combination rule that should be used."
    )

    @classmethod
    def _parameter_class(cls):
        return LennardJones612Parameter

    @classmethod
    def openmm_group(cls):
        return "NonbondedForce"

    @classmethod
    def symmetry_parameters(cls) -> List[str]:
        return ["charge", "sigma", "epsilon"]
