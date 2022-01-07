from qubekit.charges import DDECCharges, MBISCharges, SolventGaussian
from qubekit.nonbonded.protocols import MODELS, get_protocol
from qubekit.utils.datastructures import QCOptions
from qubekit.workflow.workflow import WorkFlow

# create a list of protocol definitions which can be build by a helper function
wb97xd = QCOptions(method="wB97X-D", basis="6-311++G(d,p)", program="gaussian")

b3lyp = QCOptions(method="B3LYP-D3BJ", basis="DGDZVP", program="gaussian")


ddec6 = DDECCharges(
    ddec_version=6,
)

model_0 = {
    "qc_options": wb97xd,
    "charges": ddec6,
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="0"),
}

model_1a = {
    "qc_options": b3lyp,
    "charges": ddec6,
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="1a"),
}

model_1b = {
    "qc_options": QCOptions(method="HF", basis="6-31G(d)", program="gaussian"),
    "charges": DDECCharges(ddec_version=6, solvent_settings=None),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="1b"),
}

model_2a = {
    "qc_options": wb97xd,
    "charges": DDECCharges(ddec_version=6, solvent_settings=SolventGaussian(epsilon=2)),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="2a"),
}

model_2b = {
    "qc_options": wb97xd,
    "charges": DDECCharges(
        ddec_version=6, solvent_settings=SolventGaussian(epsilon=10)
    ),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="2b"),
}

model_2c = {
    "qc_options": wb97xd,
    "charges": DDECCharges(
        ddec_version=6, solvent_settings=SolventGaussian(epsilon=20)
    ),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="2c"),
}

model_3a = {
    "qc_options": wb97xd,
    "charges": DDECCharges(ddec_version=3),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="3a"),
}

model_3b = {
    "qc_options": QCOptions(method="B3LYP-D3BJ", basis="DZVP", program="psi4"),
    "charges": MBISCharges(),
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="3b"),
}

model_4a = {
    "qc_options": wb97xd,
    "charges": ddec6,
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="4a"),
}

model_4b = {
    "qc_options": wb97xd,
    "charges": ddec6,
    "virtual_sites": None,
    "non_bonded": get_protocol(protocol_name="4b"),
}

model_5a = {
    "qc_options": wb97xd,
    "charges": ddec6,
    "non_bonded": get_protocol(protocol_name="5a"),
}

model_5b = {
    "qc_options": wb97xd,
    "charges": ddec6,
    "non_bonded": get_protocol(protocol_name="5b"),
}

model_5c = {
    "qc_options": wb97xd,
    "charges": DDECCharges(ddec_version=3),
    "non_bonded": get_protocol(protocol_name="5c"),
}

model_5d = {
    "qc_options": b3lyp,
    "charges": ddec6,
    "non_bonded": get_protocol(protocol_name="5d"),
}

model_5e = {
    "qc_options": QCOptions(method="B3LYP-D3BJ", basis="DZVP", program="psi4"),
    "charges": MBISCharges(),
    "non_bonded": get_protocol(protocol_name="5e"),
}

workflow_protocols = {
    "0": model_0,
    "1a": model_1a,
    "1b": model_1b,
    "2a": model_2a,
    "2b": model_2b,
    "2c": model_2c,
    "3a": model_3a,
    "3b": model_3b,
    "4a": model_4a,
    "4b": model_4b,
    "5a": model_5a,
    "5b": model_5b,
    "5c": model_5c,
    "5d": model_5d,
    "5e": model_5e,
}


def get_workflow_protocol(workflow_protocol: MODELS) -> WorkFlow:
    """
    Get a workflow via its protocol alias.
    """
    return WorkFlow.parse_obj(workflow_protocols[workflow_protocol])
