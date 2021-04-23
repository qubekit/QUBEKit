import copy
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm
from pydantic import Field
from torsiondrive import td_api
from typing_extensions import Literal

from qubekit.engines.geometry_optimiser import GeometryOptimiser
from qubekit.molecules import TorsionData, TorsionDriveData
from qubekit.utils import constants
from qubekit.utils.datastructures import GridPointResult, SchemaBase, TorsionScan
from qubekit.utils.file_handling import folder_setup
from qubekit.utils.helpers import export_torsiondrive_data

if TYPE_CHECKING:
    from qubekit.molecules import Ligand
    from qubekit.utils.datastructures import LocalResource, QCOptions


class TorsionDriver(SchemaBase):

    type: Literal["torsiondriver"] = "torsiondriver"
    n_workers: int = Field(
        1,
        description="The number of workers that should be used to run parallel optimisations. Note the threads and memory will be divided between workers.",
        ge=1,
        le=4,
    )
    grid_spacing: int = Field(
        15, description="The grid spacing in degrees between each optimisation."
    )
    energy_decrease_thresh: Optional[float] = Field(
        None,
        description="Threshold of an energy decrease to trigger activate new grid point. Default is 1e-5",
    )
    energy_upper_limit: Optional[float] = Field(
        None,
        description="Upper limit of energy relative to current global minimum to spawn new optimization tasks.",
    )
    starting_conformations: int = Field(
        1,
        description="The number of starting conformations that should be used in the torsiondrive. Note for a molecule with multipule flexible bonds you may need to sample them all.",
    )

    def run_torsiondrive(
        self,
        molecule: "Ligand",
        dihedral_data: TorsionScan,
        qc_spec: "QCOptions",
        local_options: "LocalResource",
    ) -> "Ligand":
        """
        Run a torsion drive for the given molecule and the targeted dihedral. The results of the scan are packed into the
        ligand into the qm_scans.

        Args:
            molecule: The ligand to be scanned.
            dihedral_data: The dihedral that has been targeted in the ligand and the scan range.

        Returns:
            The molecule with the results of the scan saved in it.
        """
        td_state = self._create_initial_state(
            molecule=molecule, dihedral_data=dihedral_data, qc_spec=qc_spec
        )
        return self._run_torsiondrive(
            td_state=td_state,
            molecule=molecule,
            qc_spec=qc_spec,
            local_options=local_options,
        )

    def _run_torsiondrive(
        self,
        td_state: Dict[str, Any],
        molecule: "Ligand",
        qc_spec: "QCOptions",
        local_options: "LocalResource",
    ) -> "Ligand":
        """
        The main torsiondrive control function.

        we assume that the settings have already been validated.

        Args:
            td_state: The initial/current torsiondrive state object.
            molecule: The target molecule we will be torsion driving.
        """

        # build the geometry optimiser
        geometry_optimiser = self._build_geometry_optimiser()
        # create a new local resource object by dividing the current one by n workers
        resource_settings = local_options.divide_resource(n_tasks=self.n_workers)
        complete = False
        target_dihedral = td_state["dihedrals"][0]
        while not complete:
            new_jobs = self._get_new_jobs(td_state=td_state)
            if not new_jobs:
                # new jobs returns an empty dict when complete
                break

            work_list = []
            results = []
            if self.n_workers > 1:
                # start worker pool for multiple optimisers
                with Pool(processes=self.n_workers) as pool:
                    print(
                        f"setting up {self.n_workers} workers to compute optimisations"
                    )
                    for grid_id_str, job_geo_list in new_jobs.items():
                        for geo_job in job_geo_list:
                            work_list.append(
                                pool.apply_async(
                                    func=optimise_grid_point,
                                    args=(
                                        geometry_optimiser,
                                        molecule,
                                        qc_spec,
                                        resource_settings,
                                        geo_job,
                                        target_dihedral,
                                        grid_id_str,
                                    ),
                                )
                            )

                    for work in tqdm.tqdm(
                        work_list,
                        total=len(work_list),
                        ncols=80,
                        desc="Optimising grid points",
                    ):
                        result = work.get()
                        # now we need to store the results
                        results.append(result)
            else:
                # make a work list as well
                for grid_id_str, job_geo_list in new_jobs.items():
                    for geo_job in job_geo_list:
                        work_list.append(
                            (
                                geometry_optimiser,
                                molecule,
                                qc_spec,
                                # use the full local options
                                local_options,
                                geo_job,
                                target_dihedral,
                                grid_id_str,
                            )
                        )
                for work in tqdm.tqdm(
                    work_list,
                    total=len(work_list),
                    ncols=80,
                    desc="Optimising grid points",
                ):
                    result = optimise_grid_point(*work)
                    results.append(result)

            # now update the state with results
            self._update_state(td_state=td_state, result_data=results)

        return self._collect_results(td_state=td_state, molecule=molecule)

    def _collect_results(
        self, td_state: Dict[str, Any], molecule: "Ligand"
    ) -> "Ligand":
        """
        After the torsiondrive has been completed collect the results and pack them into the ligand.
        """
        # torsiondrive will only give us the lowest energy for each grid point
        # we have to then work out what the final geometry is from this
        optimised_energies = td_api.collect_lowest_energies(td_state=td_state)
        final_grid = td_state["grid_status"]
        # make a torsiondrive data store assuming 1D only
        torsion_data = TorsionDriveData(
            grid_spacing=self.grid_spacing,
            dihedral=td_state["dihedrals"][0],
            torsion_drive_range=td_state["dihedral_ranges"][0],
        )
        # now grab each grid point in sorted order
        for (
            angle,
            energy,
        ) in sorted(optimised_energies.items()):
            # grab all optimisation done at this angle
            optimisations = final_grid[str(angle[0])]
            # loop over each result and check if the energy matches the lowest
            # results -> (initial geometry, final geometry, final energy)
            for result in optimisations:
                if result[-1] == energy:
                    grid_data = TorsionData(
                        angle=angle[0],
                        geometry=np.array(result[1]) * constants.BOHR_TO_ANGS,
                        energy=energy,
                    )
                    torsion_data.add_grid_point(grid_data=grid_data)
                    break
        # validate the data
        torsion_data.validate_angles()
        # dump to file
        export_torsiondrive_data(molecule=molecule, tdrive_data=torsion_data)
        # save to mol
        molecule.add_qm_scan(scan_data=torsion_data)
        # dump the torsiondrive state to file
        td_api.current_state_json_dump(td_state, "torsiondrive_state.json")

        return molecule

    def _create_initial_state(
        self,
        molecule: "Ligand",
        dihedral_data: TorsionScan,
        qc_spec: "QCOptions",
    ) -> Dict[str, Any]:
        """
        Create the initial state for the torsion drive using the input settings.

        Note:
            We also put the running spec into the state.

        Args:
            molecule: The molecule we want to create our initial state for.
            dihedral_data: The dihedral and scan range information.

        Returns:
            The torsiondrive dict or the initial state.
        """

        coords = copy.deepcopy(molecule.coordinates)
        td_state = td_api.create_initial_state(
            dihedrals=[
                dihedral_data.torsion,
            ],
            grid_spacing=[
                self.grid_spacing,
            ],
            elements=[atom.atomic_symbol for atom in molecule.atoms],
            init_coords=[(coords * constants.ANGS_TO_BOHR)],
            dihedral_ranges=[
                dihedral_data.scan_range,
            ],
            energy_decrease_thresh=self.energy_decrease_thresh,
            energy_upper_limit=self.energy_upper_limit,
        )
        td_state["spec"] = {
            "program": qc_spec.program.lower(),
            "method": qc_spec.method.lower(),
            "basis": qc_spec.basis.lower()
            if qc_spec.basis is not None
            else qc_spec.basis,
        }
        return td_state

    def _build_geometry_optimiser(self) -> GeometryOptimiser:
        """
        Build a geometry optimiser using the specified options.
        """

        geom = GeometryOptimiser(
            # set to be gau as this is default for torsiondrives
            convergence="GAU",
        )
        return geom

    def _get_new_jobs(self, td_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the job queue with new jobs from the torsiondrive API.

        Args:
            td_state: The current torsiondrive state object.

        Important:
            The coordinates returned here are in bohr.
        """
        new_jobs = td_api.next_jobs_from_state(td_state=td_state, verbose=True)
        return new_jobs

    def _update_state(
        self, td_state: Dict[str, Any], result_data: List[GridPointResult]
    ) -> Dict[str, Any]:
        """
        Update the td_state with the result queue and save to file.

        Note:
            Torsiondrive api always wants the coordinates to be an np.array

        Args:
            td_state: The current torsiondrive state object.
            result_data: The list of result data from the current round of optimisations.

        Returns:
            An updated torsiondrive state object.
        """
        job_results = {}
        for result in result_data:
            job_results.setdefault(str(result.dihedral_angle), []).append(
                (
                    np.array(result.input_geometry),
                    np.array(result.final_geometry),
                    result.final_energy,
                )
            )
        # now update the state with results
        td_api.update_state(td_state=td_state, job_results=job_results)
        # save to file
        td_api.current_state_json_dump(td_state, "torsiondrive_state.json")

        return td_state


def _build_optimiser_settings(
    dihedral: Tuple[int, int, int, int], dihedral_angle: float
) -> Dict[str, Any]:
    """
    Build up the optimiser settings dict which will be merged into the keywords to control the optimisation.
    """
    constraints = {
        "set": [{"type": "dihedral", "indices": dihedral, "value": dihedral_angle}]
    }
    optimiser_extras = {
        "coordsys": "dlc",
        "reset": True,
        "qccnv": True,
        "convergence_set": "gau",
        "constraints": constraints,
        "enforce": 0.1,
        "epsilon": 0.0,
    }
    return optimiser_extras


def optimise_grid_point(
    geometry_optimiser: GeometryOptimiser,
    molecule: "Ligand",
    qc_spec: "QCOptions",
    local_options: "LocalResource",
    # coordinates in bohr
    coordinates: List[float],
    dihedral: Tuple[int, int, int, int],
    dihedral_angle: int,
) -> GridPointResult:
    """
    For the given molecule at its initial coordinates perform a restrained optimisation at the given dihedral angle.

    This is separated from the class to make multiprocessing lighter.

    Args:
        geometry_optimiser: The geometry optimiser that should be used, this should already be configured to the correct method and basis.
        molecule: The molecule which is to be optimised.
        coordinates: The input coordinates in bohr made by torsiondrive.
        dihedral: The atom indices of the dihedral which should be fixed.
        dihedral_angle: The angle the dihedral should be set to during the optimisation.

    Returns:
        The result of the optimisation which contains the initial and final geometry along with the final energy.
    """
    # build a folder to run the calculation in we only store the last calculation at the grid point.
    with folder_setup(folder_name=f"grid_point_{dihedral_angle}"):
        # build the optimiser constraints and set torsiondrive settings
        optimiser_settings = _build_optimiser_settings(
            dihedral=dihedral, dihedral_angle=dihedral_angle
        )
        opt_mol = copy.deepcopy(molecule)
        input_coords = np.array(coordinates)
        opt_mol.coordinates = (input_coords * constants.BOHR_TO_ANGS).reshape(
            (opt_mol.n_atoms, 3)
        )
        result_mol, full_result = geometry_optimiser.optimise(
            molecule=opt_mol,
            qc_spec=qc_spec,
            local_options=local_options,
            allow_fail=False,
            return_result=True,
            extras=optimiser_settings,
        )
        # make the result class
        result_data = GridPointResult(
            dihedral_angle=dihedral_angle,
            input_geometry=coordinates,
            final_geometry=(result_mol.coordinates * constants.ANGS_TO_BOHR)
            .ravel()
            .tolist(),
            final_energy=full_result.energies[-1],
        )
        return result_data
