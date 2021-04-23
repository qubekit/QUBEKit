from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Literal

from qubekit.engines import GeometryOptimiser, call_qcengine
from qubekit.utils.datastructures import LocalResource, QCOptions, StageBase
from qubekit.utils.exceptions import HessianCalculationFailed
from qubekit.utils.helpers import check_symmetry

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


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

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
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
        )
        with open("result.json", "w") as output:
            output.write(result.json(indent=2))

        if not result.success:
            raise HessianCalculationFailed(
                "The hessian calculation failed please check the result json."
            )

        np.savetxt("hessian.txt", result.return_result)
        molecule.hessian = result.return_result
        check_symmetry(molecule.hessian)
        return molecule


class QMOptimise(StageBase):
    """A helper class to run the QM optimisation stage."""

    type: Literal["QMOptimise"] = "QMOptimise"

    @classmethod
    def is_available(cls) -> bool:
        # The spec is pre-validated so we should not fail here
        return True

    def start_message(self, **kwargs) -> str:
        qc_spec: QCOptions = kwargs["qc_spec"]
        return f"Optimising the molecule using {qc_spec.program} with {qc_spec.method}/{qc_spec.basis}."

    def finish_message(self, **kwargs) -> str:
        return "Molecule optimisation complete."

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Run a molecule geometry optimisation to the gau_tight criteria and store the final geometry into the molecule.
        """
        pass
