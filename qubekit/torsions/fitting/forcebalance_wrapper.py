"""
Classes that help with parameter fitting using ForceBalance.
"""
import abc
import os
import socket
import subprocess
from typing import Any, Dict, List

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.molecules import Ligand, TorsionDriveData
from qubekit.torsions.utils import forcebalance_setup
from qubekit.utils.datastructures import StageBase
from qubekit.utils.exceptions import ForceBalanceError, MissingReferenceData
from qubekit.utils.file_handling import get_data, make_and_change_into
from qubekit.utils.helpers import export_torsiondrive_data


class Priors(BaseModel):
    """
    A class which controls the forcebalance force field prior values.
    """

    Proper_k: float = Field(
        6.0, description="The initial prior for the proper torsion k values."
    )

    def format_priors(self) -> Dict[str, Any]:
        """
        Returns:
            A formatted dict version of the prior that can be consumed by forcebalance.
        """
        data = {}
        for prior, value in self.__dict__.items():
            prior = prior.split("_")
            prior = "/".join(prior)
            data[prior] = value
        return data


class TargetBase(BaseModel, abc.ABC):
    """
    A base class which each forcebalnce target should overwrite.
    This should control the run time settings used during fitting and implement a file setup method which is called before fitting.
    """

    target_name: str
    description: str
    writelevel: PositiveInt = 2
    keywords: Dict[str, Any] = {}
    openmm_platform: Literal["Reference"] = "Reference"

    class Config:
        validate_assignment = True
        fields = {
            "target_name": {
                "description": "The name of the forcebalance target to be fit"
            },
            "entries": {"description": "a list of target entries to be optimised."},
            "writelevel": {
                "description": "The write level controls the types of intermedate information which is saved from an optimisation."
            },
            "keywords": {
                "description": "Any keyword information which should be passed to forcebalance as a run time setting."
            },
            "openmm_platform": {
                "description": "The openmm platform that should be used in the fitting."
            },
        }

    @abc.abstractmethod
    def prep_for_fitting(self, molecule: Ligand) -> None:
        """
        The target will need to convert the input reference data to some format ready for fitting, this method should be implimented to take
        each molecule and assume it has the correct reference data.
        """
        ...

    def fb_options(self) -> Dict[str, Any]:
        """
        Format the target class run time options into a dict that can be consumed by forcebalance.
        """
        data = self.dict(exclude={"target_name", "description", "keywords"})
        data.update(self.keywords)
        return data


class TorsionProfileSmirnoff(TargetBase):
    """
    This helps set up the files required to perform the torsion profile fitting target in forcebalance.
    For each ligand passed the input files are prepared for each target torsion, the optimize in file is also updated with the target info.
    """

    target_name: Literal["TorsionProfile_SMIRNOFF"] = "TorsionProfile_SMIRNOFF"
    description = "Relaxed energy and RMSD fitting for torsion drives only."
    energy_denom: PositiveFloat = Field(
        1.0,
        description="The energy denominator used by forcebalance to weight the energies contribution to the objective function.",
    )
    energy_upper: PositiveFloat = Field(
        10.0,
        description="The upper limit for energy differences in kcal/mol which are included in fitting. Relative energies above this value do not contribute to the objective.",
    )
    attenuate: bool = Field(
        False,
        description="If the weights should be attenuated as a function of the energy above the minimum.",
    )
    restrain_k: float = Field(
        1.0,
        description="The strength of the harmonic restraint in kcal/mol used in the mm relaxation on all non-torsion atoms.",
    )
    keywords: Dict[str, Any] = {}

    def fb_options(self) -> Dict[str, Any]:
        data = super(TorsionProfileSmirnoff, self).fb_options()
        # add the specific options for this target
        data["pdb"] = "molecule.pdb"
        # ForceBalance is able to read an .sdf even though we pass it as a mol2 file.
        # This is a workaround not having a .mol2 writer (only OpenEye works fine now)
        data["mol2"] = "molecule.sdf"
        data["coords"] = "scan.xyz"
        return data

    def prep_for_fitting(self, molecule: Ligand) -> List[str]:
        """
        For the given ligand prep the input files ready for torsion profile fitting.

        Args:
            molecule: The molecule object that we need to prep for fitting, this should have qm reference data stored in molecule.qm_scans.

        Note:
            We assume we are already in the targets folder.

        Returns:
            A list of target folder names made by this target.

        Raises:
            MissingReferenceData: If the molecule does not have any torsion drive reference data saved in molecule.qm_scans.
        """
        # make sure we have data
        if not molecule.qm_scans:
            raise MissingReferenceData(
                f"Can not prepare a forcebalance fitting target for {molecule.name} as the reference data is missing!"
            )

        # write out the qdata and other input files for each scan
        target_folders = []
        # keep track of where we start
        base_folder = os.getcwd()

        # loop over each fragment and then each scanned bond and make a target folder for each
        for scan in molecule.qm_scans:
            task_name = f"{self.target_name}_{molecule.name}_{scan.central_bond[0]}_{scan.central_bond[1]}"
            target_folders.append(task_name)
            make_and_change_into(name=task_name)
            molecule.to_file(file_name="molecule.pdb")
            # mol2 is required for smirks
            molecule.to_file("molecule.sdf")
            # write the qdata file
            export_torsiondrive_data(molecule=molecule, tdrive_data=scan)
            # make the metadata
            self.make_metadata(torsiondrive_data=scan)
            # now move back to the base
            os.chdir(base_folder)

        return target_folders

    @staticmethod
    def make_metadata(torsiondrive_data: TorsionDriveData) -> None:
        """
        Create the metadata.json required to run a torsion profile target, this details the constrained optimisations to be done.

        Args:
            torsiondrive_data: Create the torsion metadata file which describes the angle to be scanned and the dihedral grid points.
        """
        import json

        json_data = {
            "dihedrals": [
                torsiondrive_data.dihedral,
            ],
            "grid_spacing": [
                torsiondrive_data.grid_spacing,
            ],
            "dihedral_ranges": [
                torsiondrive_data.torsion_drive_range,
            ],
            "torsion_grid_ids": [
                [
                    data.angle,
                ]
                for data in torsiondrive_data.reference_data.values()
            ],
        }
        # now dump to file
        with open("metadata.json", "w") as meta:
            meta.write(json.dumps(json_data, indent=2))


class ForceBalanceFitting(StageBase):
    """
    This class interfaces with forcebalance <https://github.com/leeping/forcebalance> and allows users to fit multiple force field parameters
    to multiple different targets such as optimised geometries vibration frequencies and relaxed torsion profiles.

    Note:
        We only support relaxed torsion profile fitting so far.
        All targets are fitted using OpenMM.
    """

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    type: Literal["ForceBalanceFitting"] = "ForceBalanceFitting"
    penalty_type: Literal["L1", "L2"] = "L1"
    job_type: str = "optimize"
    max_iterations: PositiveInt = 10
    convergence_step_criteria: PositiveFloat = 0.01
    convergence_objective_criteria: PositiveFloat = 0.01
    convergence_gradient_criteria: PositiveFloat = 0.01
    n_criteria: PositiveInt = 1
    eig_lowerbound: PositiveFloat = 0.01
    finite_difference_h: PositiveFloat = 0.01
    penalty_additive: PositiveFloat = 0.1
    constrain_charge: bool = False
    initial_trust_radius: float = -0.25
    minimum_trust_radius: float = 0.05
    error_tolerance: PositiveFloat = 1.0
    adaptive_factor: PositiveFloat = 0.2
    adaptive_damping: PositiveFloat = 1.0
    normalize_weights: bool = False
    extras: Dict[str, Any] = {}
    priors: Priors = Priors()
    targets: Dict[str, TorsionProfileSmirnoff] = {
        "TorsionProfile_OpenFF": TorsionProfileSmirnoff()
    }

    def start_message(self, **kwargs) -> str:
        return "Performing torsion optimisations using ForceBalance."

    def finish_message(self, **kwargs) -> str:
        return "Torsion optimisation complete."

    @classmethod
    def is_available(cls) -> bool:
        """Make sure forcebalance can be imported."""
        fb = which_import(
            "forcebalance",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install forcebalance -c conda-forge`.",
        )
        openmm = which_import(
            "openmm",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install openmm -c conda-forge`.",
        )
        return fb and openmm

    def run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        return self._run(molecule, *args, **kwargs)

    def _run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        """
        The main run method of the ForceBalance torsion optimisation stage.

        Args:
            molecule: The molecule that should be optimised with its QM reference data.

        Important:
            We work on a copy of the molecule as we have to change some parameters.
        """
        # check we have targets to fit
        if not self.targets:
            raise ForceBalanceError(
                "No fitting targets have been set for forcebalance, please set at least one target."
            )

        # now we have validated the data run the optimiser
        return self._optimise(molecule=molecule, local_options=kwargs['local_options'])

    def add_target(self, target: TargetBase) -> None:
        """
        Try and add the given target class to the forcebalance optimiser to be executed when optimise is called.

        Args:
            target: The ForceBalance optimisation target that should be added.
        """
        if issubclass(type(target), TargetBase):
            self.targets[target.target_name] = target

    def _optimise(self, molecule: Ligand, local_options: "LocalResource") -> Ligand:
        """
        For the given input molecule run the forcebalance fitting for the list of targets and run time settings.

        Note:
            The list of optimisation targets should be set before running.
        """

        # set up the master fitting folder
        with forcebalance_setup(folder_name="ForceBalance"):
            fitting_folder = os.getcwd()
            fitting_targets = {}
            # prep the target folders
            os.chdir("targets")
            molecules_to_optimise = []
            if molecule.qm_scans is not None:
                molecules_to_optimise.append(molecule)
            if molecule.fragments is not None:
                molecules_to_optimise.extend(molecule.fragments)
            for target in self.targets.values():

                for mol in molecules_to_optimise:
                    target_folders = target.prep_for_fitting(molecule=mol)
                    # fitting_targets[target.target_name] = target_folders
                    if target.target_name in fitting_targets:
                        fitting_targets[target.target_name].extend(target_folders)
                    else:
                        fitting_targets[target.target_name] = target_folders

            # back to fitting folder
            os.chdir(fitting_folder)
            # now we can make the optimize in file
            wq_port = self.generate_optimise_in(target_data=fitting_targets)
            # now make the forcefield file
            self.generate_forcefield(molecule=molecule)

            # now execute forcebalance
            with open("log.txt", "w") as log:
                forcebalance_process = subprocess.Popen(['ForceBalance', 'optimize.in'], stdout=log, stderr=log)

                import work_queue
                workers = work_queue.Factory("local", manager_host_port=f"localhost:{wq_port}")
                # the optimisation takes space in gas, so 1 core for each OpenMM is good enough most likely
                workers.cores = 1
                # divide the memory for each worker
                workers.memory = local_options.memory / local_options[0].cores
                workers.min_workers = 1
                workers.max_workers = local_options.cores

                with workers:
                    forcebalance_process.wait()

            result_ligand = self.collect_results(molecule=molecule)
            return result_ligand

    def generate_forcefield(self, molecule: Ligand) -> None:
        """
        For the given molecule generate the fitting forcefield with the target torsion terms tagged with the parameterize keyword.
        The selection of dihedrals for parameterization relies on the .fragments and their central bonds.

        Args:
            molecule: The molecule whose torsion parameters should be optimised.

        Note:
            We currently hard code to only fit dihedrals that pass through the targeted rotatable bond.

        Important:
            We work with a copy of the molecule here as we add attributes to the forcefield and change default values.
        """

        # the atoms in the dihedrals for parameterization have mass 0
        # so the constraints have to be turned off
        molecule._optimizeable_offxml(
            file_name=os.path.join("forcefield", "bespoke.offxml"), h_constraints=False
        )

    def generate_optimise_in(self, target_data: Dict[str, List[str]]) -> None:
        """
        For the given list of targets and entries produce an optimize.in file which contains all of the run time settings to be used in the optimization.
        this uses jinja templates to generate the required file from the template distributed with qubekit.

        Args:
            target_data: A dictionary mapping the target name to the target folders that have been created.
        """
        from jinja2 import Template

        template_file = get_data(os.path.join("templates", "optimize.txt"))
        with open(template_file) as file:
            template = Template(file.read())

        data = self.dict(exclude={"targets", "priors"})
        data["priors"] = self.priors.format_priors()
        data["fitting_targets"] = target_data
        target_options = {}
        for target in self.targets.values():
            target_options[target.target_name] = target.fb_options()
        data["target_options"] = target_options

        # select an empty socket and bind it
        sock = socket.socket()
        sock.bind(('', 0))
        wq_port = sock.getsockname()[1]
        data['wq_port'] = wq_port

        rendered_template = template.render(**data)

        with open("optimize.in", "w") as opt_in:
            opt_in.write(rendered_template)

        return wq_port

    @staticmethod
    def check_converged() -> bool:
        """
        Read the output from a forcebalance run to determine the exit status of the optimisation.

        Returns:
            `True` if the optimisation has converged else `False`
        """
        converged = False
        with open("optimize.out") as log:
            for line in log.readlines():
                if "optimization converged" in line.lower():
                    converged = True
                    break
                elif "convergence failure" in line.lower():
                    converged = False
                    break

            return converged

    def collect_results(self, molecule: Ligand) -> Ligand:
        """
        Collect the results of an optimisation by checking the exit status and then transferring the optimised parameters from the final
        xml forcefield back into the ligand.

        Args:
            molecule: The molecule that was optimised and where the parameters should be stored.

        Returns:
            The input molecule with the optimised parameters saved.

        Raises:
            ForceBalanceError: If the optimisation did not converge or exit properly.
        """
        # first check the exit status
        status = self.check_converged()
        if not status:
            raise ForceBalanceError(
                f"The optimisation for molecule {molecule.name} did not converge so the parameters could not be updated."
            )

        else:
            from openff.toolkit.typing.engines.smirnoff import ForceField
            from openmm import unit

            optimized_ff = ForceField(
                os.path.join("result", self.job_type, "bespoke.offxml"),
                load_plugins=True,
                allow_cosmetic_attributes=True,
            )
            # only be the torsions which were updated so extract and update the parameters
            torsion_handler = optimized_ff.get_parameter_handler("ProperTorsions")
            for parameter in torsion_handler.parameters:
                if parameter.attribute_is_cosmetic("parameterize"):
                    parameter_data = {
                        "k1": parameter.k1.value_in_unit(unit.kilojoule_per_mole),
                        "k2": parameter.k2.value_in_unit(unit.kilojoule_per_mole),
                        "k3": parameter.k3.value_in_unit(unit.kilojoule_per_mole),
                        "k4": parameter.k4.value_in_unit(unit.kilojoule_per_mole),
                        "periodicity1": parameter.periodicity1,
                        "periodicity2": parameter.periodicity2,
                        "periodicity3": parameter.periodicity3,
                        "periodicity4": parameter.periodicity4,
                        "phase1": parameter.phase1.value_in_unit(unit.radian),
                        "phase2": parameter.phase2.value_in_unit(unit.radian),
                        "phase3": parameter.phase3.value_in_unit(unit.radian),
                        "phase4": parameter.phase4.value_in_unit(unit.radian),
                    }
                    # this torsion was optimised update the parameters
                    matches = molecule.get_smarts_matches(parameter.smirks)
                    for match in matches:
                        molecule.TorsionForce[match].update(**parameter_data)

            return molecule
