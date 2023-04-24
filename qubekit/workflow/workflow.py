import os
from datetime import datetime
from typing import ClassVar, List, Optional, Union

from pydantic import Field, PrivateAttr
from typing_extensions import Literal

import qubekit
from qubekit.bonded import ModSeminario, QForceHessianFitting
from qubekit.charges import DDECCharges, MBISCharges
from qubekit.fragmentation import WBOFragmentation
from qubekit.molecules import Ligand
from qubekit.nonbonded import LennardJones612, VirtualSites, get_protocol
from qubekit.parametrisation import XML, AnteChamber, OpenFF
from qubekit.torsions import ForceBalanceFitting, TorsionScan1D
from qubekit.utils.datastructures import LocalResource, QCOptions, SchemaBase, StageBase
from qubekit.utils.exceptions import MissingReferenceData, WorkFlowExecutionError
from qubekit.utils.file_handling import folder_setup, make_and_change_into
from qubekit.workflow.helper_stages import Hessian, Optimiser
from qubekit.workflow.results import StageResult, Status, WorkFlowResult


class WorkFlow(SchemaBase):
    type: Literal["WorkFlow"] = "WorkFlow"
    qc_options: QCOptions = Field(
        QCOptions(),
        description="The QC options to be used for all QC calculations apart from implicit solvent.",
    )
    local_resources: LocalResource = Field(
        LocalResource(),
        description="The local resource options for the workflow like total memory and cores available.",
    )
    fragmentation: Optional[WBOFragmentation] = Field(
        WBOFragmentation(),
        description="The fragmentation method which should be used to speed up the torsion optimisation stage.",
    )
    parametrisation: Union[OpenFF, XML, AnteChamber] = Field(
        OpenFF(),
        description="The parametrisation engine which should be used to assign initial parameters.",
    )
    optimisation: Optimiser = Field(
        Optimiser(),
        description="The main geometry optimiser settings including pre_optimisation settings.",
    )
    charges: Union[DDECCharges, MBISCharges] = Field(
        DDECCharges(),
        description="The method that should be used to calculate the AIM reference data the charge should be extracted from. Note that the non-bonded parameters are also calculated from this data.",
    )
    virtual_sites: Optional[VirtualSites] = Field(
        VirtualSites(),
        description="The method that should be used to fit virtual sites if they are requested.",
    )
    non_bonded: LennardJones612 = Field(
        get_protocol(protocol_name="0"),
        description="The method that should be used to calculate the non-bonded non-charge parameters and their functional form.",
    )
    bonded_parameters: Union[ModSeminario, QForceHessianFitting] = Field(
        ModSeminario(),
        description="The method that should be used to optimise the bonded parameters.",
    )
    torsion_scanner: TorsionScan1D = Field(
        TorsionScan1D(),
        description="The method that should be used to drive the torsions",
    )
    torsion_optimisation: ForceBalanceFitting = Field(
        ForceBalanceFitting(),
        description="The method that should be used to optimise the scanned soft dihedrals.",
    )
    hessian: ClassVar[Hessian] = Hessian()

    _results_fname: str = PrivateAttr("workflow_result.json")

    _default_workflow: ClassVar[List[str]] = [
        "fragmentation",
        "parametrisation",
        "optimisation",
        "hessian",
        "charges",
        "virtual_sites",
        "non_bonded",
        "bonded_parameters",
        "torsion_scanner",
        "torsion_optimisation",
    ]

    @classmethod
    def from_results(cls, results: WorkFlowResult):
        """Build a workflow from the provenance info in the results object."""
        model_data = {
            "qc_options": results.qc_spec.dict(),
            "local_resources": results.local_resources.dict(),
        }
        # now loop over the stages and update the options
        for stage_name, result in results.results.items():
            if stage_name != "Hessian":  # this stage has no settings
                model_data[stage_name] = result.stage_settings
        return cls(**model_data)

    def validate_workflow(
        self, workflow: List[str], molecule: Optional[Ligand] = None
    ) -> None:
        """
        Make sure that the workflow can be run ahead of time by looking for errors in the QCspecification and missing dependencies.

        Args:
            workflow: The list of stages to be run which should be checked.
        """
        # first check the general qc spec
        self.qc_options.validate_specification()
        # then check each component for missing dependencies
        for field in workflow:
            stage = getattr(self, field)
            # some stages are optional and should be skipped
            if stage is not None:
                stage.is_available()
        # check special stages
        # check that the pre_opt method is available
        if "optimisation" in workflow:
            stage = self.optimisation
            if stage.pre_optimisation_method is not None:
                pre_spec = stage.convert_pre_opt(method=stage.pre_optimisation_method)
                pre_spec.validate_specification()
        # if we are doing nonbonded check the element coverage
        if "non_bonded" in workflow and molecule is not None:
            self.non_bonded.check_element_coverage(molecule=molecule)

    def to_file(self, filename: str) -> None:
        """
        Write the workflow to file supported file types are json or yaml.
        """
        f_type = filename.split(".")[-1]
        with open(filename, "w") as output:
            if f_type in ["yaml" or "yml"]:
                import yaml

                output.write(yaml.dump(self.dict()))
            else:
                output.write(self.json(indent=2))

    @classmethod
    def get_running_order(
        cls,
        start: Optional[str] = None,
        skip_stages: Optional[List[str]] = None,
        end: Optional[str] = None,
    ) -> List[str]:
        """Work out the running order based on any skip stages and the end point.

        Args:
            start: The starting stage for the workflow.
            skip_stages: A list of stages to remove from the workflow.
            end: The final stage which should be executed.

        Returns:
            A list of stage names in the order they will be ran in.
        """
        normal_workflow = cls._default_workflow.copy()

        if skip_stages is not None:
            for stage in skip_stages:
                try:
                    normal_workflow.remove(stage)
                except ValueError:
                    continue
        if start is not None:
            start_id = normal_workflow.index(start)
        else:
            start_id = None
        if end is not None:
            end_id = normal_workflow.index(end) + 1
        else:
            end_id = None

        return normal_workflow[start_id:end_id]

    def _build_initial_results(self, molecule: Ligand) -> WorkFlowResult:
        """Build the initial results schema using the workflow."""
        workflow = self.get_running_order()
        result = WorkFlowResult(
            version=qubekit.__version__,
            input_molecule=molecule.copy(deep=True),
            qc_spec=self.qc_options,
            current_molecule=molecule,
            local_resources=self.local_resources,
        )
        # for each stage set if they are to be ran
        for stage_name in workflow:
            stage: Optional[StageBase] = getattr(self, stage_name)
            if stage is not None:
                stage_result = StageResult(
                    stage=stage.type, stage_settings=stage.dict(), status=Status.Waiting
                )
            else:
                stage_result = StageResult(
                    stage=None, stage_settings=None, status=Status.Waiting
                )

            result.results[stage_name] = stage_result
        return result

    def _get_optional_stage_skip(
        self, skip_stages: Optional[List[str]]
    ) -> Optional[List[str]]:
        """
        Add any optional stages which are skipped when not supplied, to the skip stages list.
        """

        optional_stages = []
        for stage_name in self._default_workflow:
            stage = getattr(self, stage_name)
            if stage is None:
                optional_stages.append(stage_name)

        if optional_stages and skip_stages is not None:
            # we get a tuple from click so we can not append
            return [*skip_stages, *optional_stages]

        elif optional_stages and skip_stages is None:
            return [*optional_stages]

        return skip_stages

    def restart_workflow(
        self,
        start: str,
        result: WorkFlowResult,
        skip_stages: Optional[List[str]] = None,
        end: Optional[str] = None,
    ) -> WorkFlowResult:
        """
        Restart the workflow from the given stage and continue the run.

        Args:
            start: The name of the stage we want to restart the workflow from.
            result: The past run results object which will be updated and that starting molecule will be extracted from.
            skip_stages: The list of stage names which should be skipped.
            end: The name of the last stage to be computed before finishing.
        """
        molecule = result.current_molecule
        skip_stages = self._get_optional_stage_skip(skip_stages=skip_stages)
        run_order = self.get_running_order(
            start=start, skip_stages=skip_stages, end=end
        )
        # update local and qc options
        result.qc_spec = self.qc_options
        result.local_resources = self.local_resources
        return self._run_workflow(molecule=molecule, results=result, workflow=run_order)

    def new_workflow(
        self,
        molecule: Ligand,
        skip_stages: Optional[List[str]] = None,
        end: Optional[str] = None,
    ) -> WorkFlowResult:
        """
        The main worker method to be used for starting new workflows.

        Args:
            molecule: The molecule to be re-parametrised using QUBEKit.
            skip_stages: A list of stages which should be skipped in the workflow.
            end: The last stage to be computed before finishing, useful to finish early.
        """
        # get the running order
        skip_stages = self._get_optional_stage_skip(skip_stages=skip_stages)
        run_order = self.get_running_order(skip_stages=skip_stages, end=end)
        results = self._build_initial_results(molecule=molecule)
        # if we have any skips assign them
        for stage_name, stage_result in results.results.items():
            if stage_name not in run_order:
                stage_result.status = Status.Skip

        return self._run_workflow(
            molecule=molecule, workflow=run_order, results=results
        )

    def _run_workflow(
        self,
        molecule: "Ligand",
        workflow: List[str],
        results: WorkFlowResult,
    ) -> WorkFlowResult:
        """
        The main run method of the workflow which will execute each stage inorder on the given input molecule.

        Args:
            molecule: The molecule to be re-parametrised using QUBEKit.
            workflow: The list of prefiltered stage names which should be ran in order.
            results: The results object that we should update throughout the workflow.

        Returns:
            A fully parametrised molecule.
        """
        # try and find missing dependencies
        self.validate_workflow(workflow=workflow, molecule=molecule)
        # start message
        # TODO Move to outside workflow so this doesn't get printed for every run in bulk.
        print(
            "If QUBEKit ever breaks or you would like to view timings and loads of other info, "
            "view the log file.\nOur documentation (README.md) "
            "also contains help on handling the various commands for QUBEKit.\n"
        )
        # write out the results object to track the status at the start
        results.to_file(filename=self._results_fname)

        # loop over stages and run
        for field in workflow:
            stage: StageBase = getattr(self, field)
            # some stages print based on what spec they are using
            print(stage.start_message(qc_spec=self.qc_options, molecule=molecule))
            molecule = self._run_stage(
                stage_name=field, stage=stage, molecule=molecule, results=results
            )
            print(stage.finish_message(molecule=molecule))

        # now the workflow has finished
        # write final results
        results.to_file(filename=self._results_fname)
        # write out final parameters
        with folder_setup("final_parameters"):
            molecule.to_file(file_name=f"{molecule.name}.pdb")
            molecule.to_file(file_name=f"{molecule.name}.sdf")
            molecule.write_parameters(file_name=f"{molecule.name}.xml")
            # write out the offxml with h-bond constraints
            molecule.to_offxml(file_name=f"{molecule.name}.offxml", h_constraints=True)
            # write without h-constraints
            molecule.to_offxml(
                file_name=f"{molecule.name}_unconstrained.offxml", h_constraints=False
            )

        return results

    def _run_stage(
        self,
        stage_name: str,
        stage: StageBase,
        molecule: Ligand,
        results: WorkFlowResult,
    ) -> Ligand:
        """
        A stage wrapper to run the stage and update the results workflow in place.
        """
        home = os.getcwd()
        # update settings and set to running and save
        stage_result = StageResult(
            stage=stage.type, stage_settings=stage.dict(), status=Status.Running
        )
        results.results[stage_name] = stage_result
        results.modified_date = datetime.now().strftime("%Y_%m_%d")
        results.to_file(filename=self._results_fname)
        make_and_change_into(name=stage_name)
        try:
            # run the stage and save the result
            result_mol = stage.run(
                molecule=molecule,
                qc_spec=self.qc_options,
                local_options=self.local_resources,
            )
            stage_result.status = Status.Done
            results.current_molecule = result_mol
        except MissingReferenceData:
            # this means there are no torsions to scan so simulate it working
            stage_result.status = Status.Done
            results.current_molecule = molecule
        except Exception as e:
            import traceback

            # save the error do not update the current molecule
            stage_result.status = Status.Error
            stage_result.error = traceback.extract_tb(e.__traceback__).format()
            os.chdir(home)
            results.results[stage_name] = stage_result
            results.to_file(self._results_fname)
            # write the exception to file
            with open("QUBEKit.err", "w") as output:
                traceback.print_exc(file=output)

            raise WorkFlowExecutionError(
                f"The workflow stopped unexpectedly due to the following error at stage: {stage_name}"
            ) from e

        # move back
        os.chdir(home)
        # update the results
        results.results[stage_name] = stage_result
        results.to_file(self._results_fname)
        return results.current_molecule
