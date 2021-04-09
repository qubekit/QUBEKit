#!/usr/bin/env python3

from typing import Dict, Optional

import qcelemental as qcel
import qcengine as qcng
from pydantic import BaseModel, PositiveInt, validator

from qubekit.utils.exceptions import SpecificationError


class BaseEngine(BaseModel):
    """
    A common base model for engines to run computations via external programs.
    """

    program: str = "psi4"
    basis: Optional[str] = "6-311++G(d,p)"
    method: str = "wB97X-D"
    cores: PositiveInt = 4
    memory: PositiveInt = 10

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        fields = {
            "program": {
                "description": "The name of the program which should be used to carry out the computation, such as psi4"
            },
            "basis": {
                "description": "The basis that should be used in the computation."
            },
            "method": {
                "description": "The method that should be used for the computation."
            },
            "cores": {
                "description": "The number of cores to be allocated to the computation."
            },
            "memory": {
                "description": "The amount of memory that should be allocated to the computation in GB."
            },
        }

    @validator("cores", "memory")
    def validate_resource(cls, resource: int) -> int:
        """
        Make sure that the resource is even to make distribution easier.
        """
        if resource % 2 != 0 or resource != 1:
            raise ValueError(
                f"The resource must be even to make distribution between parallel workers easier."
            )
        return resource

    @validator("program")
    def validate_program(cls, program: str) -> str:
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
