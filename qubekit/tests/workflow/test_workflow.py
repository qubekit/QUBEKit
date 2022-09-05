import pytest

from qubekit.workflow import WorkFlow, get_workflow_protocol


def test_running_order_double_skip():
    """
    Dont raise an error if we try and skip the same step twice.
    """
    running_order = WorkFlow.get_running_order(
        skip_stages=["hessian", "hessian", "charges"]
    )
    assert "hessian" not in running_order
    assert "charges" not in running_order


def test_running_order_start():
    """
    Make sure the running order correctly starts at the requested stage.
    """
    running_order = WorkFlow.get_running_order(start="charges")
    assert len(running_order) == 6
    assert running_order[0] == "charges"


def test_running_order_end():
    """
    Make sure the end stage is included when supplied to the running order.
    """
    running_order = WorkFlow.get_running_order(end="charges")
    assert len(running_order) == 5
    assert running_order[-1] == "charges"


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("workflow.json", id="json"),
        pytest.param("workflow.yaml", id="yaml"),
    ],
)
def test_to_file(filename, tmpdir):
    """
    Test writing to file.
    """
    model = get_workflow_protocol(workflow_protocol="0")
    with tmpdir.as_cwd():
        model.to_file(filename=filename)


def test_validate_workflow(acetone):
    """
    Make sure workflow validation correctly checks the specification and each stage.
    """
    model0 = get_workflow_protocol(workflow_protocol="0")
    # change to an rdkit spec which is available always
    model0.qc_options.program = "rdkit"
    model0.qc_options.method = "uff"
    model0.qc_options.basis = None
    # assuming no psi4 or gaussian
    run_order = WorkFlow.get_running_order(skip_stages=["charges"])
    model0.validate_workflow(workflow=run_order, molecule=acetone)


def test_build_results(acetone):
    """
    Make sure we can build a results object from the workflow which captures the correct settings.
    """
    # get a model with all stages present
    model5a = get_workflow_protocol(workflow_protocol="5a")
    result = model5a._build_initial_results(molecule=acetone)
    assert result.input_molecule.json() == acetone.json()
    for result in result.results.values():
        assert result.status == "waiting"


@pytest.mark.parametrize(
    "workflow_model", [pytest.param("0", id="0"), pytest.param("5a", id="5a")]
)
def test_results_round_trip(acetone, workflow_model):
    """
    Test round tripping a workflow to and from results.
    """
    workflow = get_workflow_protocol(workflow_protocol=workflow_model)
    results = workflow._build_initial_results(molecule=acetone)
    workflow_2 = WorkFlow.from_results(results=results)
    assert workflow.json() == workflow_2.json()


def test_build_results_optional(acetone):
    """
    Test building a results object with an optional stage.
    """
    model0 = get_workflow_protocol(workflow_protocol="0")
    result = model0._build_initial_results(molecule=acetone)
    assert result.results["virtual_sites"].stage_settings is None


@pytest.mark.parametrize(
    "skip_stages",
    [pytest.param(["charges", "hessian"], id="stages"), pytest.param(None, id="None")],
)
def test_optional_stage_skip(skip_stages):
    """
    Test adding the optional stages to the skip list.
    """
    # has no vsite stage
    model0 = get_workflow_protocol(workflow_protocol="0")
    skips = model0._get_optional_stage_skip(skip_stages=skip_stages)
    assert "virtual_sites" in skips


def test_run_workflow(acetone, tmpdir, rdkit_workflow):
    """
    Run a very simple workflow on only the parameterisation stage.
    """
    with tmpdir.as_cwd():
        result = rdkit_workflow.new_workflow(molecule=acetone, end="parametrisation")
        assert result.results["parametrisation"].status == "done"
        assert result.results["parametrisation"].error is None


def test_restart_workflow(acetone, tmpdir, rdkit_workflow):
    """
    Make sure workflows can be restarted from the results objects.
    """

    with tmpdir.as_cwd():
        result = rdkit_workflow.new_workflow(molecule=acetone, end="parametrisation")
        assert result.results["parametrisation"].status == "done"
        assert result.results["optimisation"].status == "skip"
        result = rdkit_workflow.restart_workflow(
            start="optimisation", result=result, end="optimisation"
        )
        assert result.results["optimisation"].status == "done"
