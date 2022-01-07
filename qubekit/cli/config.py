from typing import Optional

import click

from qubekit.workflow import WorkFlow, WorkFlowResult, get_workflow_protocol


def prep_config(
    config_file: Optional[str] = None,
    results: Optional[WorkFlowResult] = None,
    cores: Optional[int] = None,
    memory: Optional[int] = None,
    protocol: Optional[str] = None,
) -> WorkFlow:
    """A helper function to load the config file for the CLI and update common options.
    Note:
        If not config file path is given then we load the default workflow.
    """
    if config_file is not None and protocol is not None:
        raise RuntimeError(
            "The `protocol` and `config` options are mutually exclusive please only supply one."
        )

    # load the local config
    if config_file is not None:
        workflow = WorkFlow.parse_file(config_file)
    # load config from results
    elif results is not None:
        workflow = WorkFlow.from_results(results=results)
    elif protocol is not None:
        workflow = get_workflow_protocol(workflow_protocol=protocol)
    else:
        # use the basic workflow if no config given, model 0
        workflow = get_workflow_protocol(workflow_protocol="0")

    # update the workflow
    workflow.local_resources.cores = cores or workflow.local_resources.cores
    workflow.local_resources.memory = memory or workflow.local_resources.memory
    return workflow


protocols = click.Choice(
    [
        "0",
        "1a",
        "1b",
        "2a",
        "2b",
        "2c",
        "3a",
        "3b",
        "4a",
        "4b",
        "5a",
        "5b",
        "5c",
        "5d",
        "5e",
    ],
    case_sensitive=True,
)


@click.group()
def config():
    """Make and validate workflows."""
    pass


@config.command()
@click.argument("filename", type=click.STRING)
@click.option(
    "-p",
    "--protocol",
    type=protocols,
    help="The alias of the parametrisation protocol.",
    default="0",
)
def create(filename: str, protocol: str) -> None:
    """Create a new config file using the standard workflow and write to file."""
    w = get_workflow_protocol(workflow_protocol=protocol)
    w.to_file(filename=filename)


@config.command()
@click.argument("filename", type=click.Path(exists=True))
def validate(filename: str) -> None:
    """Validate a config file and make sure there are no missing dependencies."""
    w = WorkFlow.parse_file(path=filename)
    # validate the full workflow
    workflow = w.get_running_order()
    w.validate_workflow(workflow=workflow)
