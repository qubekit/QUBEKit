import pytest
from click.testing import CliRunner


@pytest.fixture(scope="module")
def run_cli() -> CliRunner:
    """
    Create a new click CLI runner.
    """
    runner = CliRunner()

    with runner.isolated_filesystem():
        yield runner
