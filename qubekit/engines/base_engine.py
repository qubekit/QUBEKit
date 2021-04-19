#!/usr/bin/env python3

from typing import Dict, Optional

import qcelemental as qcel
import qcengine as qcng
from pydantic import Field, PositiveInt, validator

from qubekit.utils.datastructures import SchemaBase
from qubekit.utils.exceptions import SpecificationError


class BaseEngine(SchemaBase):
    """
    A common base model for engines to run computations via external programs.
    """

    program: str = Field(
        "psi4",
        description="The name of the program which should be used to carry out the computation, such as psi4",
    )
    basis: Optional[str] = Field(
        "6-311++G(d,p)", description="The basis that should be used in the computation."
    )
    method: str = Field(
        "wB97X-D", description="The method that should be used for the computation."
    )
    cores: PositiveInt = Field(
        4, description="The number of cores to be allocated to the computation."
    )
    memory: PositiveInt = Field(
        10,
        description="The amount of memory that should be allocated to the computation in GB.",
    )

    @validator("program")
    def _validate_program(cls, program: str) -> str:
        """
        Validate the choice of program against those supported by QCEngine and QUBEKit.
        """
        programs = qcng.list_available_programs()
        programs.discard("dftd3")

        if program.lower() not in programs:
            raise SpecificationError(
                f"The program {program} is not available, available programs are {programs}"
            )
        return program.lower()

    @property
    def local_options(self) -> Dict[str, int]:
        """return the local options."""
        return {"memory": self.memory, "ncores": self.cores}

    @property
    def qc_model(self) -> qcel.models.common_models.Model:
        """
        Build the QC model for the computation.
        """
        model = qcel.models.common_models.Model(method=self.method, basis=self.basis)
        return model


class Engines:
    """
    Engines base class containing core information that all other engines
    (PSI4, Gaussian etc) will have.
    """

    def __init__(self, molecule: "Ligand"):

        self.molecule = molecule

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"
