from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from qubekit.molecules import Ligand
from qubekit.utils.datastructures import LocalResource, QCOptions, SchemaBase


class Status(str, Enum):
    Done = "done"
    Error = "error"
    Running = "running"
    Skip = "skip"
    Waiting = "waiting"


class StageResult(SchemaBase):
    """
    A results class to hold basic information about the progress of the stage.
    """

    stage: Optional[str] = Field(
        ...,
        description="The stage identifier i.e stage.type as this is not the same as the general workflow stage name.",
    )
    stage_settings: Optional[Dict[str, Any]] = Field(
        ..., description="The dict of the stage serialising any settings used."
    )
    status: Status = Field(
        ..., description="If the stage successfully completed `True` or not `False`."
    )
    error: Optional[Any] = Field(
        None,
        description="If the stage was not successful the error message should be stored here.",
    )


class WorkFlowResult(SchemaBase):
    """
    A results schema which details the workflow used on the input molecule, and the status of each stage along with
    a molecule checkpoint.
    """

    version: str = Field(
        ..., description="The version of QUBEKit used to run the workflow."
    )
    creation_date: str = Field(
        datetime.now().strftime("%Y_%m_%d"),
        description="The initial date the workflow was ran on.",
    )
    modified_date: str = Field(
        datetime.now().strftime("%Y_%m_%d"),
        description="The last date and time the results schema was modified on.",
    )
    input_molecule: Ligand = Field(
        ..., description="The input molecule given to the workflow."
    )
    qc_spec: QCOptions = Field(
        ...,
        description="The main QC specification used to calculate the bonded parameters.",
    )
    local_resources: LocalResource = Field(
        ..., description="The local resource used to run the workflow."
    )
    current_molecule: Ligand = Field(
        ...,
        description="The checkpoint molecule with the latest result of the workflow. When restarting this molecule will be used as input.",
    )
    results: Dict[str, StageResult] = Field(
        {},
        description="The result from each stage is stored here and is updated each time work flow is updated.",
    )

    def to_file(self, filename: str) -> None:
        """
        Write the workflow to file supported file types are json.
        """
        with open(filename, "w") as output:
            output.write(self.json(indent=2))

    def status(self) -> Dict[str, Status]:
        """
        Get the current status of the workflow.
        """
        return dict(
            (stage_name, stage.status) for stage_name, stage in self.results.items()
        )
