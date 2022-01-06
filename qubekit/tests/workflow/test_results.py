import pytest


def test_results_to_file(acetone, rdkit_workflow, tmpdir):
    """
    Test writing a results object to file.
    """
    with tmpdir.as_cwd():
        result = rdkit_workflow._build_initial_results(molecule=acetone)
        result.to_file(filename="result.json")


def test_result_status(acetone, rdkit_workflow):
    """
    Make sure the status of each stage is collected correctly.
    """

    result = rdkit_workflow._build_initial_results(molecule=acetone)
    all_status = result.status()
    for status in all_status.values():
        assert status == "waiting"
