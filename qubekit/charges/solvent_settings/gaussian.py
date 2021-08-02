from typing import Dict, List

from pydantic import Field
from typing_extensions import Literal

from qubekit.charges.solvent_settings.base import SolventBase


class SolventGaussian(SolventBase):
    """A simple schema to encode the Gaussian implicit solvent settings.

    Important:
        All Rfree values were fit to IPCM with an epsilon of 4. IPCM is not available with TD-SCF calculations
        and so PCM must be used. This is still in development.
    """

    program: Literal["gaussian"] = "gaussian"
    solver_type: Literal["IPCM", "PCM"] = Field(
        "IPCM",
        description="The solver type to be used when calculating the polarised density.",
    )
    epsilon: float = Field(
        4.0,
        description="The epsilon value of the implicit solvent as we do not used a parametrised solvent.",
        gt=0,
    )
    volume_contour: float = Field(
        0.0004,
        description="The contour value the volume is defined inside of in electrons/bohr^3 used to write out the density.",
    )

    def format_keywords(self) -> Dict[str, List[str]]:
        """
        Format the options into a dict that can be consumed by the gaussian harness.
        """
        # this depends on the solver used
        eps_setting = (
            f"Eps={self.epsilon}"
            if self.solver_type == "PCM"
            else f"\n{self.epsilon} {self.volume_contour}"
        )
        data = dict(
            cmdline_extra=[
                f"SCRF=({self.solver_type}, Read)",
                "density=current",
                "OUTPUT=WFX",
            ],
            add_input=[eps_setting, "gaussian.wfx"],
        )
        return data
