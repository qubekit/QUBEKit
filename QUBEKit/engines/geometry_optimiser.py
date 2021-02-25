"""
A class which handles general geometry optimisation tasks.
"""
from typing import Optional, Dict, Union
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator, PositiveInt
from QUBEKit.utils.exceptions import SpecificationError
import qcelemental as qcel
import qcengine as qcng


class GeometryOptimiser(BaseModel):
    """
    A general geometry optimiser class which can dispatch the optimisation call to the correct program and method via qcengine.
    #TODO do we want to expose more optimiser settings?
    """

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    optimiser: Literal["geometric", "optking", "native"] = Field(
        "geometric",
        description="The name of the optimisation engine which should be used note only gaussian supports native optimisation.",
    )
    program: str = Field(
        "openmm",
        description="The name of the program which should be used to run the optimisation, for a full list see QCEngine.",
    )
    basis: Optional[str] = Field(
        "smirnoff", description="The basis that should be used during the optimisation."
    )
    method: str = Field(
        "openff_unconstrained-1.3.0.offxml",
        description="The name of the method that should be used to run the optimisation.",
    )
    maxiter: PositiveInt = Field(
        350, description="The maximum number of optimisation steps."
    )
    convergence: Literal["GAU", "GAU_TIGHT"] = Field(
        "GAU_TIGHT",
        description="The convergence critera for the geometry optimisation.",
    )
    cores: int = Field(4, description="The number of cores to use in the optimisation")
    memory: int = Field(
        4, description="The amount of memory in GB the program can use."
    )
    extras: Optional[Dict] = Field(
        None,
        description="Any extra arguments that should be passed to the geometry optimiser, like scf maxiter.",
    )

    @validator("program")
    def validate_program(cls, program: str) -> str:
        """
        Validate the choice of program against those supported by QCEngine and QUBEKit.
        """
        from qcengine import list_available_programs

        programs = list_available_programs()
        programs.discard("dftd3")
        programs.add("gaussian")

        if program.lower() not in programs:
            raise SpecificationError(
                f"The program {program} is not available, available programs are {programs}"
            )
        return program.lower()

    @property
    def local_options(self) -> Dict[str, int]:
        """return the local options."""
        return {"memory": self.memory, "ncores": self.cores}

    def _validate_specification(self) -> None:
        """
        Validate the specification this is called before running an optimisation to catch errors before run time.
        """
        from openff.toolkit.typing.engines.smirnoff import get_available_force_fields

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
        if self.program.lower() == "gaussian" and self.optimiser != "native":
            raise SpecificationError(
                f"The program gaussian currently only supports its native optimiser."
            )

        # we do not validate QM as there are so many options
        if self.program.lower() in settings:
            program_settings = settings[self.program.lower()]

            allowed_methods = program_settings.get(self.basis, set())
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

    def __init__(
        self,
        optimiser,
        program="openmm",
        basis="smirnoff",
        method="openff",
        maxiter=320,
        convergence="GAU",
    ):
        """
        Validate.
        """
        super().__init__(
            optimiser=optimiser,
            program=program,
            basis=basis,
            method=method,
            maxiter=maxiter,
            convergence=convergence,
        )
        self._validate_specification()

    def build_optimiser_keywords(self) -> Dict[str, Union[str, float]]:
        """
        Based on the selected optimiser return the optimisation keywords.
        Are there any other options we want to expose?
        """
        opt_keywords = {"program": self.program}
        if self.optimiser == "geometric":
            opt_keywords["coordsys"] = "dlc"
            opt_keywords["maxiter"] = self.maxiter
            opt_keywords["convergence_set"] = self.convergence
        elif self.optimiser == "optking":
            opt_keywords["geom_maxiter"] = self.maxiter
            opt_keywords["G_CONVERGENCE"] = self.convergence
        return opt_keywords

    @staticmethod
    def check_available(program: str, optimiser: str) -> bool:
        """
        A wrapper around the general harness check in qcengine,
        here we can check that a optimiser and program are installed before run time.
        """
        _ = qcng.get_program(name=program, check=True)
        _ = qcng.get_procedure(name=optimiser)
        return True

    def optimise(
        self, molecule: "Ligand", input_type: str = "input"
    ) -> qcel.models.OptimizationResult:
        """
        For the given specification in the class run an optimisation on the ligand.
        """
        # first validate the settings
        self._validate_specification()
        # now validate that the programs are installed
        self.check_available(program=self.program, optimiser=self.optimiser)

        # now we need to distribute the job
        model = qcel.models.common_models.Model(method=self.method, basis=self.basis)
        specification = qcel.models.procedures.QCInputSpecification(model=model)
        initial_mol = molecule.to_qcschema(input_type=input_type)
        opt_task = qcel.models.OptimizationInput(
            initial_molecule=initial_mol,
            input_specification=specification,
            keywords=self.build_optimiser_keywords(),
        )
        opt_result = qcng.compute_procedure(
            input_data=opt_task,
            procedure=self.optimiser,
            raise_error=True,
            local_options=self.local_options,
        )
        return opt_result
