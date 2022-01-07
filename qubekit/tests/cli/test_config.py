import pytest

from qubekit.cli.config import create, prep_config, validate
from qubekit.utils.exceptions import SpecificationError
from qubekit.workflow import WorkFlow, get_workflow_protocol


def test_prep_config_mutual():
    """
    Make sure an error is raised if we pass mutually exclusive options.
    """

    with pytest.raises(RuntimeError, match="The `protocol` and `config` options"):
        prep_config(config_file="workflow.json", protocol="0")


def test_prep_config_default():
    """
    Make sure we get protocol 0 by default when no args are past.
    """

    workflow = prep_config()

    assert workflow.virtual_sites is None


def test_prep_config_local_options():
    """
    Make sure any local options are used over defaults
    """
    workflow = prep_config(memory=100, cores=50)

    assert workflow.local_resources.cores == 50
    assert workflow.local_resources.memory == 100


def test_config_prep_protocol():
    """
    Make sure we get the correct protocol for the alias we pass.
    """
    # pick an non-default workflow
    workflow = prep_config(protocol="5e")
    assert workflow.virtual_sites is not None
    assert workflow.qc_options.method == "b3lyp-d3bj"
    assert workflow.qc_options.program == "psi4"
    assert workflow.charges.type == "MBISCharges"
    assert workflow.charges.solvent_settings.medium_Solvent == "CHCL3"


def test_config_prep_file(tmpdir):
    """
    Make sure we can load a config file.
    """
    with tmpdir.as_cwd():
        workflow_0 = get_workflow_protocol(workflow_protocol="0")
        workflow_0.to_file(filename="workflow.json")
        workflow = prep_config(config_file="workflow.json")
        assert workflow.json() == workflow_0.json()


def test_config_create_cli(tmpdir, run_cli):
    """
    Test creating a config from the cli using a protocol
    """

    output = run_cli.invoke(create, args=["-p", "5c", "workflow.json"])
    assert output.exit_code == 0
    workflow = WorkFlow.parse_file("workflow.json")
    assert workflow.charges.ddec_version == 3


def test_validate(tmpdir, run_cli):
    """
    Test validating a workflow via the cli.
    """

    workflow = get_workflow_protocol(workflow_protocol="0")
    workflow.to_file(filename="workflow.json")
    output = run_cli.invoke(validate, args=["workflow.json"])
    # gaussian is not installed so make sure we have the correct error
    assert output.exit_code == 1
    assert isinstance(output.exception, SpecificationError)
