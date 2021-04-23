from typing import ClassVar, List, Optional, TypeVar, Union

from pydantic import Field
from typing_extensions import Literal

from qubekit.charges import DDECCharges, MBISCharges
from qubekit.engines import GeometryOptimiser, call_qcengine
from qubekit.mod_seminario import ModSeminario
from qubekit.nonbonded import LennardJones612
from qubekit.parametrisation import XML, AnteChamber, OpenFF
from qubekit.torsions import ForceBalanceFitting, TorsionScan1D
from qubekit.utils.datastructures import LocalResource, QCOptions, SchemaBase, StageBase
from qubekit.virtual_sites import VirtualSites
from qubekit.workflow.helper_stages import Hessian, QMOptimise


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
    parametrisation: Union[OpenFF, XML, AnteChamber] = Field(
        OpenFF(),
        description="The parametrisation engine which should be used to assign initial parameters.",
    )
    bonded_parameters: ModSeminario = Field(
        ModSeminario(symmetrise_parameters=True),
        description="The method that should be used to optimise the bonded parameters.",
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
        LennardJones612(),
        description="The method that should be used to calculate the non-bonded non-charge parameters and their functional form.",
    )
    torsion_scanner: TorsionScan1D = Field(
        TorsionScan1D(),
        description="The method that should be used to drive the torsions",
    )
    torsion_optimisation: ForceBalanceFitting = Field(
        ForceBalanceFitting(),
        description="The method that should be used to optimise the scanned soft dihedrals.",
    )
    qm_optimise: ClassVar[QMOptimise] = Field(
        QMOptimise(),
        description="A helper stage which will run a geometry optimisation using QM starting from the current conformer.",
    )
    hessian: ClassVar[Hessian] = Field(
        Hessian(),
        description="A stage which will run a hessian calculation on the molecule at the current geometry.",
    )

    def validate_workflow(self) -> None:
        """
        Make sure that the workflow can be run ahead of time by looking for errors in the QCspecification and missing dependencies.
        """
        # first check the general qc spec
        self.qc_options.validate_specification()
        # then check each component for missing dependencies
        for field in self.__fields__.keys():
            stage = getattr(self, field)
            if issubclass(type(stage), StageBase):
                _ = stage.is_available()

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

    def run(self):
        for field in self.__fields__.keys():
            stage: StageBase = getattr(self, field)
            if issubclass(type(stage), StageBase):
                print(stage.start_message(qc_spec=self.qc_options))
                print(stage.finish_message())
