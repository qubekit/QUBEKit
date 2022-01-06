from datetime import datetime
from typing import List, Optional

import click

from qubekit.cli.config import prep_config
from qubekit.cli.run import runtime_options, stages
from qubekit.molecules import Ligand
from qubekit.utils.exceptions import WorkFlowExecutionError
from qubekit.utils.file_handling import folder_setup
from qubekit.workflow import WorkFlowResult


@click.group()
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
    protocol: Optional[str] = None,
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
                            results=results, cores=cores, memory=memory, protocol=None
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
                    protocol=protocol,
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
