import pytest

from qubekit.cli.run import run
from qubekit.utils.file_handling import get_data


def test_run_mutual(run_cli):
    """
    Make sure an error is raised when running the CLI with mutually exclusive args.
    """

    output = run_cli.invoke(
        run, args=["-i", get_data("acetone.sdf"), "-sm", "CC(=O)C", "-n", "acetone"]
    )
    assert output.exit_code == 1
    assert isinstance(output.exception, RuntimeError)


def test_run_no_name(run_cli):
    """
    Make sure an error is raised if we start from smiles but do not supply a name.
    """

    output = run_cli.invoke(
        run,
        args=[
            "-sm",
            "CC(=O)C",
        ],
    )
    assert output.exit_code == 1
    assert isinstance(output.exception, RuntimeError)
