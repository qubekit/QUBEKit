from datetime import datetime
from typing import Callable, List, Optional

import click

import qubekit
from qubekit.molecules import Ligand
from qubekit.utils.exceptions import WorkFlowExecutionError
from qubekit.utils.file_handling import folder_setup
from qubekit.workflow import WorkFlow, WorkFlowResult

stages = click.Choice(
    [
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


def prep_config(
    config_file: Optional[str] = None,
    results: Optional[WorkFlowResult] = None,
    cores: Optional[int] = None,
    memory: Optional[int] = None,
) -> WorkFlow:
    """A helper function to load the config file for the CLI and update common options.
    Note:
        If not config file path is given then we load the default workflow.
    """
    # load the local config
    if config_file is not None:
        workflow = WorkFlow.parse_file(config_file)
    # load config from results
    elif results is not None:
        workflow = WorkFlow.from_results(results=results)
    else:
        # use the basic workflow if no config given
        workflow = WorkFlow()

    # update the workflow
    workflow.local_resources.cores = cores or workflow.local_resources.cores
    workflow.local_resources.memory = memory or workflow.local_resources.memory
    return workflow


@click.group()
@click.version_option(version=qubekit.__version__, prog_name="QUBEKit")
def cli():
    pass


@cli.group()
def config():
    """Make and validate workflows."""
    pass


@config.command()
@click.argument("filename", type=click.STRING)
def create(filename: str) -> None:
    """Create a new config file using the standard workflow and write to file."""
    w = WorkFlow()
    w.to_file(filename=filename)


@config.command()
@click.argument("filename", type=click.Path(exists=True))
def validate(filename: str) -> None:
    """Validate a config file and make sure there are no missing dependencies."""
    w = WorkFlow.parse_file(path=filename)
    # validate the full workflow
    workflow = w.get_running_order()
    w.validate_workflow(workflow=workflow)


@cli.command()
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
@runtime_options
def run(
    input_file: Optional[str] = None,
    smiles: Optional[str] = None,
    name: Optional[str] = None,
    multiplicity: int = 1,
    end: Optional[str] = None,
    skip_stages: Optional[List[str]] = None,
    config: Optional[str] = None,
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
    workflow = prep_config(config_file=config, memory=memory, cores=cores)

    # move into the working folder and run
    with folder_setup(f"QUBEKit_{molecule.name}_{datetime.now().strftime('%Y_%m_%d')}"):
        # write the starting molecule
        molecule.to_file(file_name=f"{molecule.name}.pdb")
        workflow.new_workflow(molecule=molecule, skip_stages=skip_stages, end=end)


@cli.command()
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


@cli.group()
def bulk():
    """Create or run bulk workflows on a set of molecules."""
    pass


@bulk.command()
@click.argument(
    "bulk_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
)
@click.option(
    "-restart",
    "--restart",
    type=stages,
    help="The stage the workflow should be restarted from.",
)
@runtime_options
def run(
    bulk_file: str,
    skip_stages: Optional[List[str]] = None,
    end: Optional[str] = None,
    restart: Optional[str] = None,
    config: Optional[str] = None,
    cores: Optional[int] = None,
    memory: Optional[int] = None,
) -> None:
    """Run the QUBEKit parametrisation workflow on a collection of molecules in serial.

    Loop over the molecules in order of the CSV file.
    """
    import glob
    import os

    from qubekit.utils.helpers import mol_data_from_csv

    home = os.getcwd()
    # load all inputs
    bulk_data = mol_data_from_csv(bulk_file)

    # start main molecule loop
    for name, mol_data in bulk_data.items():
        print(f"Analysing: {name}")
        try:
            if restart is not None or mol_data["restart"] is not None:
                # we are trying to restart a run, find the folder
                # should only be one
                fname = name.split(".")[0]
                folder = glob.glob(f"QUBEKit_{fname}_*")[0]
                with folder_setup(folder):
                    results = WorkFlowResult.parse_file("workflow_result.json")
                    if config is None:
                        # if we have no new config load from results
                        workflow = prep_config(
                            results=results, cores=cores, memory=memory
                        )
                    else:
                        # load the new config file
                        workflow = prep_config(
                            config_file=config, cores=cores, memory=memory
                        )

                    workflow.restart_workflow(
                        start=restart or mol_data["restart"],
                        skip_stages=skip_stages,
                        end=end or mol_data["end"],
                        result=results,
                    )

            else:
                if mol_data["smiles"] is not None:
                    molecule = Ligand.from_smiles(
                        smiles_string=mol_data["smiles"], name=name
                    )
                else:
                    molecule = Ligand.from_file(file_name=name)

                # load the CLI config or the csv config, else default
                workflow = prep_config(
                    config_file=config or mol_data["config_file"],
                    memory=memory,
                    cores=cores,
                )
                # move into the working folder and run
                with folder_setup(
                    f"QUBEKit_{molecule.name}_{datetime.now().strftime('%Y_%m_%d')}"
                ):
                    # write the starting molecule
                    molecule.to_file(file_name=f"{molecule.name}.pdb")
                    workflow.new_workflow(
                        molecule=molecule,
                        skip_stages=skip_stages,
                        end=end or mol_data["end"],
                    )
        except WorkFlowExecutionError:
            os.chdir(home)
            print(
                f"An error was encountered while running {name} see folder for more info."
            )
            continue


@bulk.command()
@click.argument("filename", type=click.STRING)
def create(filename: str) -> None:
    """Generate a bulk run CSV file from all molecule files in the current directory."""
    from qubekit.utils.helpers import generate_bulk_csv

    if filename.split(".")[-1].lower() != "csv":
        filename = f"{filename}.csv"

    generate_bulk_csv(csv_name=filename)


@cli.command()
def progress():
    """Generate a report of the parametrisation workflow progress for a set of QUBEKit job folders."""
    import os

    from qubekit.utils.constants import COLOURS
    from qubekit.workflow import Status

    results = {}
    for root, _, files in os.walk(".", topdown=True):
        for file in files:
            if "workflow_result.json" in file and "backups" not in root:
                result = WorkFlowResult.parse_file(
                    os.path.abspath(os.path.join(root, file))
                )
                results[result.input_molecule.name] = result.status()

    if not results:
        print(
            "No QUBEKit directories with log files found. Perhaps you need to move to the parent directory."
        )
    else:
        # Sort alphabetically
        results = dict(sorted(results.items(), key=lambda item: item[0]))

        print("Displaying progress of all analyses in current directory.")
        print(f"Progress key: {COLOURS.green}\u2713{COLOURS.end} = Done;", end=" ")
        print(f"{COLOURS.blue}S{COLOURS.end} = Skipped;", end=" ")
        print(f"{COLOURS.red}E{COLOURS.end} = Error;", end=" ")
        print(f"{COLOURS.orange}R{COLOURS.end} = Running;", end=" ")
        print(f"{COLOURS.purple}~{COLOURS.end} = Queued")

        header_string = "{:15}" + "{:>10}" * 9
        print(
            header_string.format(
                "Name",
                "Param",
                "Opt",
                "Hessian",
                "Bonded",
                "Charges",
                "VSites",
                "Non-Bond",
                "Tor Scan",
                "Tor Opt",
            )
        )

        for name, result in results.items():
            print(f"{name[:13]:15}", end=" ")
            for s in result.values():
                if s == Status.Done:
                    stat = "\u2713"
                    print(f"{COLOURS.green}{stat:>9}{COLOURS.end}", end=" ")
                elif s == Status.Error:
                    stat = "E"
                    print(f"{COLOURS.red}{stat:>9}{COLOURS.end}", end=" ")
                elif s == Status.Skip:
                    stat = "S"
                    print(f"{COLOURS.blue}{stat:>9}{COLOURS.end}", end=" ")
                elif s == Status.Running:
                    stat = "R"
                    print(f"{COLOURS.orange}{stat:>9}{COLOURS.end}", end=" ")
                elif s == Status.Waiting:
                    stat = "~"
                    print(f"{COLOURS.purple}{stat:>9}{COLOURS.end}", end=" ")

            print("")


if __name__ == "__main__":
    cli()
