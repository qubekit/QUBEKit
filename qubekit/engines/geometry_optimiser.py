#!/usr/bin/env python3

"""
A class which handles general geometry optimisation tasks.
"""
import copy
from typing import Any, Dict, Optional, Tuple, Union

import qcelemental as qcel
import qcengine as qcng
from pydantic import Field, PositiveInt, validator
from typing_extensions import Literal

from qubekit.engines.base_engine import BaseEngine
from qubekit.molecules import Ligand
from qubekit.utils import constants
from qubekit.utils.exceptions import SpecificationError


class GeometryOptimiser(BaseEngine):
    """
    A general geometry optimiser class which can dispatch the optimisation call to the correct program and method via qcengine.
    #TODO do we want to expose more optimiser settings?
    """

    optimiser: str = Field(
        "geometric",
        description="The name of the optimisation engine which should be used note only gaussian supports native optimisation.",
    )
    program: str = "rdkit"
    basis: Optional[str] = None
    method: str = "mmff94"
    maxiter: PositiveInt = Field(
        350, description="The maximum number of optimisation steps."
    )
    convergence: Literal["GAU", "GAU_TIGHT", "GAU_VERYTIGHT"] = Field(
        "GAU_TIGHT",
        description="The convergence critera for the geometry optimisation.",
    )
    extras: Optional[Dict] = Field(
        None,
        description="Any extra arguments that should be passed to the geometry optimiser, like scf maxiter.",
    )

    @validator("optimiser")
    def validate_optimiser(cls, optimiser: str) -> str:
        """
        Make sure the chosen optimiser is available.
        """
        procedures = qcng.list_available_procedures()
        if optimiser.lower() not in procedures:
            raise SpecificationError(
                f"The optimiser {optimiser} is not available, available optimisers are {procedures}"
            )
        return optimiser.lower()

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
        if self.program.lower() != "psi4" and self.optimiser == "optking":
            raise SpecificationError(
                f"The optimiser optking currently only supports psi4 as the engine."
            )

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

    def __init__(self, **data):
        """
        Validate.
        """
        super().__init__(**data)
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
        self,
        molecule: Ligand,
        allow_fail: bool = False,
        return_result: bool = False,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Ligand, Optional[qcel.models.OptimizationResult]]:
        """
        For the given specification in the class run an optimisation on the ligand.

        Run the specified optimisation on the ligand the final coordinates are extracted and stored in the ligand.
        The optimisation schema is dumped to file along with the optimised geometry and the trajectory.

        Args:
            molecule:
                The molecule which should be optimised
            allow_fail:
                If we should not raise an error if the molecule fails to be optimised, this will extract the last geometry
                from the trajectory and return it.
            return_result:
                If the full result json should also be returned useful for extracting the trajectory.
            extras:
                A dictionary of extras that should be used to update the optimiser keywords.

        Returns:
            A new copy of the molecule at the optimised coordinates.
        """
        # first validate the settings
        self._validate_specification()
        # now validate that the programs are installed
        self.check_available(program=self.program, optimiser=self.optimiser)

        # now we need to distribute the job
        model = self.qc_model
        specification = qcel.models.procedures.QCInputSpecification(
            model=model, keywords={"dft_spherical_points": 590, "dft_radial_points": 99}
        )
        initial_mol = molecule.to_qcschema()
        optimiser_keywords = self.build_optimiser_keywords()
        if extras is not None:
            optimiser_keywords.update(extras)
        opt_task = qcel.models.OptimizationInput(
            initial_molecule=initial_mol,
            input_specification=specification,
            keywords=optimiser_keywords,
        )
        opt_result = qcng.compute_procedure(
            input_data=opt_task,
            procedure=self.optimiser,
            raise_error=False,
            local_options=self.local_options,
        )
        # dump info to file
        result_mol = self.handle_output(molecule=molecule, opt_output=opt_result)
        # check if we can/have failed and raise the error
        if not opt_result.success and not allow_fail:
            raise RuntimeError(
                f"{opt_result.error.error_type}: {opt_result.error.error_message}"
            )

        full_result = opt_result if return_result else None
        return result_mol, full_result

    def handle_output(
        self,
        molecule: Ligand,
        opt_output: Union[
            qcel.models.OptimizationResult, qcel.models.common_models.FailedOperation
        ],
    ) -> Ligand:
        """
        Sort the output of the optimisation depending on success.

        Take a molecule and an optimisation result or failed operation and unpack the trajectory and final geometry into
        a copy of the molecule. Also save the result to file.

        Args:
            molecule:
                The molecule that the optimisation was ran on.
            opt_output:
                The output of an optimisation which may be a valid result or a FailedOperation.

        Returns:
            A copy of the molecule with the final geometry in the trajectory saved.
        """
        result_mol = copy.deepcopy(molecule)
        # Store the entire result to file
        with open("result.json", "w") as out:
            out.write(opt_output.json())

        # Now sort the result
        if opt_output.success:
            # passed operation so act normal
            trajectory = [result.molecule for result in opt_output.trajectory]
        else:
            try:
                trajectory = [
                    qcel.models.Molecule.from_data(mol["molecule"])
                    for mol in opt_output.input_data["trajectory"]
                ]
            except KeyError:
                # this is a fatal error as no compute was performed so exit early with no traj
                return result_mol

        # TODO how do we get this in the log
        # print(f"Optimisation finished in {len(trajectory)} steps.")
        traj = [mol.geometry * constants.BOHR_TO_ANGS for mol in trajectory]
        result_mol.coordinates = traj[-1]
        # write to file
        result_mol.to_file(file_name="opt.xyz")
        result_mol.to_multiconformer_file(
            file_name="opt_trajectory.xyz", positions=traj
        )

        return result_mol
