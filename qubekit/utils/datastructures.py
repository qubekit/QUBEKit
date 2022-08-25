import abc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import qcelemental as qcel
import qcengine as qcng
from openff.toolkit.typing.engines.smirnoff import get_available_force_fields
from pydantic import BaseModel, Field, PositiveInt, dataclasses, validator
from typing_extensions import Literal

from qubekit.utils.exceptions import SpecificationError

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class SchemaBase(BaseModel):
    """A basic pydantic starting class which uses assigment validation."""

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda v: v.flatten().tolist()}


class LocalResource(SchemaBase):

    cores: PositiveInt = Field(
        4, description="The number of cores to be allocated to the computation."
    )
    memory: PositiveInt = Field(
        10,
        description="The amount of memory that should be allocated to the computation in GB.",
    )

    @property
    def local_options(self) -> Dict[str, int]:
        """return the local options."""
        return {"memory": self.memory, "ncores": self.cores}

    def divide_resource(self, n_tasks: int) -> "LocalResource":
        """
        Create a new local resource object by dividing the current one by n_tasks.

        Important:
            The resource is always rounded down to avoid over subscriptions.
        """
        if n_tasks == 1:
            return self
        else:
            cores = int(self.cores / n_tasks)
            memory = int(self.memory / n_tasks)
            return LocalResource(cores=cores, memory=memory)


class TDSettings(SchemaBase):
    """
    A schema with available options for Time-Dependent calculations.
    """

    n_states: int = Field(3, description="The number of states to solve for.")
    use_tda: bool = Field(
        False, description="If we should use the Tamm-Dancoff approximation (TDA)."
    )


class QCOptions(SchemaBase):
    """
    A simple Schema to validate QC/ML/MM runtime options.
    Note this model is locked once created to avoid validation errors.
    """

    program: str = Field(
        "gaussian",
        description="The name of the program which should be used to carry out the computation, such as psi4",
    )
    basis: Optional[str] = Field(
        "6-311++G(d,p)", description="The basis that should be used in the computation."
    )
    method: str = Field(
        "wB97X-D", description="The method that should be used for the computation."
    )
    td_settings: Optional[TDSettings] = Field(
        None,
        description="Any time dependent settings that should be used during the computation. Note not all programs support this option.",
    )

    @validator("program", "method")
    def _cast_lower(cls, parameter: str) -> str:
        """Lower the parameter to avoid validation issues."""
        return parameter.lower()

    def validate_program(self):
        """
        Validate the choice of program against those supported by QCEngine and QUBEKit.
        """
        programs = qcng.list_available_programs()
        programs.discard("dftd3")

        if self.program.lower() not in programs:
            raise SpecificationError(
                f"The program {self.program} is not available, available programs are {programs}"
            )

    @property
    def keywords(self) -> Dict[str, Union[str, int]]:
        """
        Build some keywords in a consistent way for the qcspec.
        """
        keywords = {
            "scf_type": "df",
            # make sure we always use an ultrafine grid
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        }
        if self.td_settings is not None:
            # use psi4 keyword settings to be consistent
            keywords["tdscf_states"] = self.td_settings.n_states
            keywords["tdscf_tda"] = self.td_settings.use_tda

        # work around a setting in psi4, fixes range seperated functionals
        if self.program.lower() == "psi4":
            keywords["wcombine"] = False
        return keywords

    @property
    def qc_model(self) -> qcel.models.common_models.Model:
        """
        Build the QC model for the computation.

        Important:
            The method name can be changed depending on the program used and td settings
        """
        if self.td_settings is not None and self.program == "psi4":
            # we have to add the td tag
            method = self.method
            if "td" != method.split("-")[0]:
                method = f"td-{method}"
        else:
            method = self.method
        model = qcel.models.common_models.Model(method=method, basis=self.basis)
        return model

    def validate_specification(self) -> None:
        """
        Validate the specification this should be called before using the spec to find errors.
        """
        # make sure the program is valid first then the basis method combination
        self.validate_program()

        openff_forcefields = [
            ff.split(".offxml")[0].lower() for ff in get_available_force_fields()
        ]
        # set up some models
        ani_methods = {"ani1x", "ani1ccx", "ani2x"}
        xtb_methods = {
            "gfn0-xtb",
            "gfn0xtb",
            "gfn1-xtb",
            "gfn1xtb",
            "gfn2-xtb",
            "gfn2xtb",
            "gfn-ff",
            "gfnff",
        }
        rdkit_methods = {"uff", "mmff94", "mmff94s"}
        gaff_forcefields = {
            "gaff-1.4",
            "gaff-1.8",
            "gaff-1.81",
            "gaff-2.1",
            "gaff-2.11",
        }
        settings = {
            "openmm": {"antechamber": gaff_forcefields, "smirnoff": openff_forcefields},
            "torchani": {None: ani_methods},
            "xtb": {None: xtb_methods},
            "rdkit": {None: rdkit_methods},
        }
        # now check these settings
        # TODO do we raise an error or just change at run time with a warning?
        # we do not validate QM as there are so many options
        if self.program.lower() in settings:
            program_settings = settings[self.program.lower()]

            allowed_methods = program_settings.get(self.basis, None)
            if allowed_methods is None:
                raise SpecificationError(
                    f"The Basis {self.basis} is not supported for the program {self.program} please chose from {program_settings.keys()}"
                )
            # now check the method
            method = self.method.split(".offxml")[0].lower()
            if method not in allowed_methods:
                raise SpecificationError(
                    f"The method {method} is not available for the program {self.program}  with basis {self.basis}, please chose from {allowed_methods}"
                )
        if self.td_settings is not None:
            if self.program.lower() not in ["gaussian"]:
                raise SpecificationError(
                    f"The program {self.program.lower()} does not support time-dependent calculations."
                )


class StageBase(SchemaBase, abc.ABC):

    type: Literal["base"] = "base"

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Check any dependencies to make sure that this stage is available to run."""
        ...

    def run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        """run it on the molecule and on the fragments """
        molecule = self.run(molecule, *args, **kwargs)

        # run it on the fragments
        if molecule.fragments is not None:
            molecule.fragments = [self.run(fragment, *args, fragment=True, **kwargs) for fragment in molecule.fragments]

        return molecule

    @abc.abstractmethod
    def _run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        """The main function of the stage which should perform some parametrisation and return the complete molecule."""
        ...

    @abc.abstractmethod
    def start_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that stage is starting with any important options.
        """
        ...

    @abc.abstractmethod
    def finish_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that the stage is complete and any checks that have been performed.
        """
        ...


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
