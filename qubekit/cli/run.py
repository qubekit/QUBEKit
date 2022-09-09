from datetime import datetime
from typing import Callable, List, Optional

import click

from qubekit.cli.config import prep_config, protocols
from qubekit.molecules import Ligand
from qubekit.utils.file_handling import folder_setup
from qubekit.workflow import WorkFlowResult

stages = click.Choice(
    [
        "fragmentation",
        "parametrisation",
        "optimisation",
        "hessian",
        "charges",
        "virtual_sites",
        "non_bonded",
        "bonded_parameters",
        "torsion_scanner",
        "torsion_optimisation",
    ],
    case_sensitive=True,
)


def runtime_options(function: Callable) -> Callable:
    """Wrap a CLI function with common runtime option settings."""
    function = click.option(
        "-nc",
        "--cores",
        type=click.INT,
        help="The total number of cores the workflow can use.",
    )(function)
    function = click.option(
        "-mem",
        "--memory",
        type=click.INT,
        help="The total amount of memory available in GB.",
    )(function)
    function = click.option(
        "-e",
        "--end",
        help="The name of the last stage to be run in the workflow, can be used to finish a workflow early.",
        type=stages,
    )(function)
    function = click.option(
        "-s",
        "--skip-stages",
        multiple=True,
        help="The names of any stages that should be skipped in the workflow.",
        type=stages,
    )(function)
    function = click.option(
        "-c",
        "--config",
        type=click.Path(exists=True, dir_okay=False, resolve_path=True, readable=True),
        help="The name of the config file which contains the workflow.",
    )(function)
    return function


@click.command()
@click.option(
    "-i",
    "--input-file",
    help="The name of the input file containing a molecule to be parameterised.",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, readable=True),
)
@click.option(
    "-sm",
    "--smiles",
    help="The smiles string of the molecule to be parameterised.",
    type=click.STRING,
)
@click.option(
    "-m",
    "--multiplicity",
    type=click.INT,
    help="The multiplicity of the molecule used in QM calculations.",
    default=1,
)
@click.option(
    "-n",
    "--name",
    type=click.STRING,
    help="The name of the molecule, used for fileIO and folder setup.",
)
@click.option(
    "-p",
    "--protocol",
    type=protocols,
    help="The alias of the parametrisation protocol.",
)
@runtime_options
def run(
    input_file: Optional[str] = None,
    smiles: Optional[str] = None,
    name: Optional[str] = None,
    multiplicity: int = 1,
    end: Optional[str] = None,
    skip_stages: Optional[List[str]] = None,
    config: Optional[str] = None,
    protocol: Optional[str] = None,
    cores: Optional[int] = None,
    memory: Optional[int] = None,
):
    """Run the QUBEKit parametrisation workflow on an input molecule."""
    # make sure we have an input or smiles not both
    if input_file is not None and smiles is not None:
        raise RuntimeError(
            "Please supply either the name of the input file or a smiles string not both."
        )
    # load the molecule
    if input_file is not None:
        molecule = Ligand.from_file(file_name=input_file, multiplicity=multiplicity)
    else:
        if name is None:
            raise RuntimeError(
                "Please also pass a name for the molecule when starting from smiles."
            )
        molecule = Ligand.from_smiles(
            smiles_string=smiles, name=name, multiplicity=multiplicity
        )

    # load workflow
    workflow = prep_config(
        config_file=config, memory=memory, cores=cores, protocol=protocol
    )

    # move into the working folder and run
    with folder_setup(f"QUBEKit_{molecule.name}_{datetime.now().strftime('%Y_%m_%d')}"):
        # write the starting molecule
        molecule.to_file(file_name=f"{molecule.name}.pdb")
        workflow.new_workflow(molecule=molecule, skip_stages=skip_stages, end=end)


@click.command()
@click.argument("start", type=stages)
@click.option(
    "-r",
    "--results",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    help="The results file that the workflow should be restarted from.",
    default="workflow_result.json",
)
@runtime_options
def restart(
    start: str,
    results: str,
    skip_stages: Optional[List[str]] = None,
    end: Optional[str] = None,
    config: Optional[str] = None,
    cores: Optional[int] = None,
    memory: Optional[int] = None,
) -> None:
    """Restart a QUBEKit parametrisation job from the given stage.
    Must be started from within an old workflow folder.
    """
    # try and load the results file
    results = WorkFlowResult.parse_file(results)

    if config is None:
        # if we have no new config load from results
        workflow = prep_config(results=results, cores=cores, memory=memory)
    else:
        # load the new config file
        workflow = prep_config(config_file=config, cores=cores, memory=memory)

    # now run the workflow
    workflow.restart_workflow(
        start=start, result=results, skip_stages=skip_stages, end=end
    )
