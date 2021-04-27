#!/usr/bin/env python3

from typing import Dict, Optional

import qcelemental as qcel
import qcengine as qcng
from pydantic import Field
from typing_extensions import Literal

from qubekit.engines.base_engine import BaseEngine
from qubekit.molecules import Ligand


class QCEngine(BaseEngine):
    """
    A wrapper around qcengine which calls single point calculations, this automatically converts the ligand and inputs
    to the correct format.
    TODO how do we want to support keywords.
    """

    driver: Literal["energy", "gradient", "hessian"] = Field(
        "hessian", description="The single point job type that should be done."
    )

    def call_qcengine(
        self,
        molecule: Ligand,
        extras: Optional[Dict[str, str]] = None,
    ) -> qcel.models.AtomicResult:
        """
        Calculate the requested property using qcengine for the given molecule.

        Args:
            molecule: The QUBEKit ligand that the calculation should be ran on.
            extras: Any extra calculation keywords that are program specific.

        Returns:
            The full qcelemental atomic result so any required information can be extracted.
        """
        qc_mol = molecule.to_qcschema()
        # default keywords
        keywords = {
            "scf_type": "df",
            # make sure we always use an ultrafine grid
            "dft_spherical_points": 590,
            "dft_radial_points": 99,
        }
        if extras is not None:
            keywords.update(extras)
        task = qcel.models.AtomicInput(
            molecule=qc_mol,
            driver=self.driver,
            model=self.qc_model,
            keywords=keywords,
        )

        result = qcng.compute(task, self.program, local_options=self.local_options)

        return result
