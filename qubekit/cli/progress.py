import click

from qubekit.workflow import WorkFlowResult


@click.command()
def progress():
    """Generate a report of the parametrisation workflow progress for a set of QUBEKit job folders."""
    import os

    from qubekit.utils.constants import COLOURS
    from qubekit.workflow import Status

    results = {}
    for root, _, files in os.walk("..", topdown=True):
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
