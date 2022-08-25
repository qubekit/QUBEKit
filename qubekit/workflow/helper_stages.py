from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from pydantic import Field
from tqdm import tqdm
from typing_extensions import Literal

from qubekit.engines import GeometryOptimiser, call_qcengine
from qubekit.utils.datastructures import LocalResource, QCOptions, StageBase
from qubekit.utils.exceptions import GeometryOptimisationError, HessianCalculationFailed
from qubekit.utils.file_handling import folder_setup
from qubekit.utils.helpers import check_symmetry

from qubekit.molecules import Ligand, Fragment

PreOptMethods = Literal[
    "mmff94",
    "mmff94s",
    "uff",
    "gfn1xtb",
    "gfn2xtb",
    "gfn0xtb",
    "gaff-2.11",
    "ani1x",
    "ani1ccx",
    "ani2x",
    "openff-1.3.0",
]


class Hessian(StageBase):
    """A helper class to run hessian calculations."""

    type: Literal["Hessian"] = "Hessian"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def start_message(self, **kwargs) -> str:
        return "Calculating Hessian matrix."

    def finish_message(self, **kwargs) -> str:
        return "Hessian matrix calculated and confirmed to be symmetric."

    def _run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Run a hessian calculation on the ligand at the current geometry and store the result into the molecule.
        """
        qc_spec = kwargs["qc_spec"]
        local_options = kwargs["local_options"]
        result = call_qcengine(
            molecule=molecule,
            driver="hessian",
            qc_spec=qc_spec,
            local_options=local_options,
            # we need to request the wbos for the hessian for qforce
            extras={"scf_properties": ["wiberg_lowdin_indices"]},
        )
        with open("result.json", "w") as output:
            output.write(result.json())

        if not result.success:
            raise HessianCalculationFailed(
                "The hessian calculation failed please check the result json."
            )

        np.savetxt("hessian.txt", result.return_result)
        molecule.hessian = result.return_result
        if "qcvars" in result.extras:
            molecule.wbo = result.extras["qcvars"]["WIBERG LOWDIN INDICES"]
        check_symmetry(molecule.hessian)
        return molecule


class Optimiser(StageBase):
    """A helper class to run the QM optimisation stage and any pre-optimisations.

    Important:
        This stage will run a pre optimisation to the GAU convergence for each seed conformer.
        Then we attempt to optimise each seed conformer to GAU_TIGHT(or other criteria) using full QM.
        If the conformer does not converge in 50 steps we randomly bum the coordinates and restart for another 50 steps before
        moving to the next conformer.
        If no conformer can be converged we raise an error.

    Note:
        By default psi4/gaussian calculations us an UltraFine integration grid to help with tight convergence criteria.
    """

    type: Literal["Optimiser"] = "Optimiser"
    pre_optimisation_method: Optional[PreOptMethods] = Field(
        "gfn2xtb",
        description="The pre-optimisation method that should be used before full QM.",
    )
    seed_conformers: int = Field(
        40,
        description="The number of seed conformers that we should attempt to generate to find the QM minimum.",
        gt=0,
    )
    maxiter: int = Field(
        350,
        description="The maximum number of optimisation steps allowed before failing the molecule.",
    )
    convergence_criteria: Literal["GAU_TIGHT", "GAU", "GAU_VERYTIGHT"] = Field(
        "GAU_TIGHT", description="The convergence criteria for the optimisation."
    )

    @classmethod
    def is_available(cls) -> bool:
        # The spec is pre-validated so we should not fail here
        return True

    def start_message(self, **kwargs) -> str:
        return f"Optimising the molecule using pre_optimisation method {self.pre_optimisation_method}."

    def finish_message(self, **kwargs) -> str:
        return "Molecule optimisation complete."

    def _run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Run a molecule geometry optimisation to the gau_tight criteria and store the final geometry into the molecule.
        """
        local_options = kwargs["local_options"]
        qm_spec: QCOptions = kwargs["qc_spec"]

        if 'fragment' not in kwargs:
            print('Ignoring the main molecule')
            return molecule

        # first run the pre_opt method
        if self.pre_optimisation_method is not None:
            pre_opt_spec = self.convert_pre_opt(method=self.pre_optimisation_method)
            preopt_geometries = self._pre_opt(
                molecule=molecule, qc_spec=pre_opt_spec, local_options=local_options
            )
        else:
            # just generate the seed conformers
            preopt_geometries = molecule.generate_conformers(
                n_conformers=self.seed_conformers
            )
        print("Partial optimisation complete")

        # now run the main method
        print(
            f"Optimising molecule using {qm_spec.program} with {qm_spec.method}/{qm_spec.basis}"
        )
        return self._run_qm_opt(
            molecule=molecule,
            conformers=preopt_geometries,
            qc_spec=qm_spec,
            local_options=local_options,
        )

    def _run_qm_opt(
        self,
        molecule: "Ligand",
        conformers: List[np.array],
        qc_spec: QCOptions,
        local_options: LocalResource,
    ) -> "Ligand":
        """
        Run the main QM optimisation on each of the input conformers in order and stop when we get a fully optimised structure.

        Args:
            molecule: The qubekit molecule to run the optimisation on.
            conformers: The list of pre-optimised conformers.
            qc_spec: The QCSpec used to run the QM optimisation.
            local_options: The local resource that is available for the optimisation.
        """
        opt_mol = deepcopy(molecule)
        g_opt = GeometryOptimiser(convergence=self.convergence_criteria, maxiter=50)

        for i, conformer in enumerate(
            tqdm(
                conformers, desc="Optimising conformer", total=len(conformers), ncols=80
            )
        ):
            with folder_setup(folder_name=f"conformer_{i}" + molecule.suffix()):
                # set the coords
                opt_mol.coordinates = conformer
                # errors are auto raised from the class so catch the result, and write to file
                qm_result, result = g_opt.optimise(
                    molecule=opt_mol,
                    allow_fail=True,
                    return_result=True,
                    qc_spec=qc_spec,
                    local_options=local_options,
                )
                if result.success:
                    break
                else:
                    # grab last coords and bump
                    coords = qm_result.coordinates + np.random.choice(
                        a=[0, 0.01], size=(qm_result.n_atoms, 3)
                    )
                    opt_mol.coordinates = coords
                    bump_mol, bump_result = g_opt.optimise(
                        molecule=opt_mol,
                        allow_fail=True,
                        return_result=True,
                        local_options=local_options,
                        qc_spec=qc_spec,
                    )
                    if bump_result.success:
                        qm_result = bump_mol
                        break

        else:
            raise GeometryOptimisationError(
                "No molecule conformer could be optimised to GAU TIGHT"
            )
        return qm_result

    def _pre_opt(
        self, molecule: "Ligand", qc_spec: QCOptions, local_options: LocalResource
    ) -> List[np.array]:
        """Run the pre optimisation stage and return the optimised conformers ready for QM optimisation.

        Args:
            molecule: The qubekit molecule to run the optimisation on.

        Returns:
            A list of final coordinates from the optimisations.
        """
        from multiprocessing import Pool

        g_opt = GeometryOptimiser(convergence="GAU", maxiter=self.maxiter)

        # generate the input conformations, number will include the input conform if provided
        geometries = molecule.generate_conformers(n_conformers=self.seed_conformers)
        molecule.to_multiconformer_file(
            file_name=f"starting_coords{molecule.suffix()}.xyz", positions=geometries
        )
        opt_list = []

        with Pool(processes=local_options.cores) as pool:
            results = []
            for confomer in geometries:
                opt_mol = deepcopy(molecule)
                opt_mol.coordinates = confomer
                opt_list.append(
                    pool.apply_async(
                        # make sure to divide the total resource between the number of workers
                        g_opt.optimise,
                        (
                            opt_mol,
                            qc_spec,
                            local_options.divide_resource(n_tasks=local_options.cores),
                            True,
                            True,
                        ),
                    )
                )

            results = []
            for result in tqdm(
                opt_list,
                desc=f"Optimising conformers with {self.pre_optimisation_method}",
                total=len(opt_list),
                ncols=80,
            ):
                # errors are auto raised from the class so catch the result, and write to file
                # fixme: this is where it gets stuck
                result_mol, opt_result = result.get()
                if opt_result.success:
                    # save the final energy and molecule
                    results.append((result_mol, opt_result.energies[-1]))
                else:
                    # save the molecule and final energy from the last step if it fails
                    results.append((result_mol, opt_result.input_data["energies"][-1]))

        # sort the results
        results.sort(key=lambda x: x[1])
        final_geometries = [re[0].coordinates for re in results]
        # write all conformers out
        molecule.to_multiconformer_file(
            file_name=f"final_pre_opt_coords{molecule.suffix()}.xyz", positions=final_geometries
        )
        return final_geometries

    @classmethod
    def convert_pre_opt(cls, method: str) -> QCOptions:
        """Convert the preopt method string to a qcoptions object."""
        if method in ["mmff94", "mmff94s", "uff"]:
            return QCOptions(program="rdkit", basis=None, method=method)
        elif method in ["gfn1xtb", "gfn2xtb", "gfn0xtb"]:
            return QCOptions(program="xtb", basis=None, method=method)
        elif method in ["ani1x", "ani1ccx", "ani2x"]:
            return QCOptions(program="torchani", basis=None, method=method)
        elif method == "gaff-2.11":
            return QCOptions(program="openmm", basis="antechamber", method=method)
        else:
            return QCOptions(program="openmm", basis="smirnoff", method=method)
