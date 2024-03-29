#!/usr/bin/env python3

from typing import TYPE_CHECKING, Any, Dict, Optional

import qcelemental as qcel
import qcengine as qcng
from typing_extensions import Literal

if TYPE_CHECKING:
    from qubekit.molecules import Ligand
    from qubekit.utils.datastructures import LocalResource, QCOptions


def call_qcengine(
    molecule: "Ligand",
    driver: Literal["energy", "gradient", "hessian"],
    qc_spec: "QCOptions",
    local_options: "LocalResource",
    extras: Optional[Dict[str, Any]] = None,
) -> qcel.models.AtomicResult:
    """
    Calculate the requested property using qcengine for the given molecule.

    Args:
        molecule: The QUBEKit ligand that the calculation should be ran on.
        driver: The type of calculation to be done.
        qc_spec: The qc specification which details the method and basis that should be used.
        local_options: Any runtime options that should be used such as memory and cores.
        extras: Any extra calculation keywords that are program specific.

    Returns:
        The full qcelemental atomic result so any required information can be extracted.
    """
    # validate the qc spec
    qc_spec.validate_specification()
    qc_mol = molecule.to_qcschema()
    # default keywords
    keywords = qc_spec.keywords
    if extras is not None:
        keywords.update(extras)
    task = qcel.models.AtomicInput(
        molecule=qc_mol,
        driver=driver,
        model=qc_spec.qc_model,
        keywords=keywords,
    )

    result = qcng.compute(
        task, qc_spec.program, task_config=local_options.local_options
    )
    with open("qcengine_result.json", "w") as output:
        output.write(result.json())

    return result
