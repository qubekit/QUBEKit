from typing import Dict, List

from pydantic import Field
from typing_extensions import Literal

from qubekit.utils.datastructures import SchemaBase


class SolventGaussian(SchemaBase):
    """A simple schema to encode the Gaussian implicit solvent settings.

    Important:
        We currently only support the IPCM with an epsilon value of 4
    """

    solver_type: Literal["IPCM"] = Field(
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
        data = dict(
            cmdline_extra=[
                f"SCRF=({self.solver_type}, Read)",
                "density=current",
                "OUTPUT=WFX",
            ],
            add_input=[f"\n{self.epsilon} {self.volume_contour}", "gaussian.wfx"],
        )
        return data
