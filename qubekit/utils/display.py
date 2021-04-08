#!/usr/bin/env python3

import os
from collections import OrderedDict

from qubekit.utils.constants import COLOURS
from qubekit.utils.helpers import unpickle


def pretty_progress():
    """
    Neatly displays the state of all QUBEKit running directories in the terminal.
    Uses the log files to automatically generate a table which is then printed to screen in full colour 4k.
    """

    # Find the path of all files starting with QUBEKit_log and add their full path to log_files list
    log_files = []
    for root, _, files in os.walk(".", topdown=True):
        for file in files:
            if "QUBEKit_log.txt" in file and "backups" not in root:
                log_files.append(os.path.abspath(f"{root}/{file}"))

    if not log_files:
        print(
            "No QUBEKit directories with log files found. Perhaps you need to move to the parent directory."
        )
        return

    # Open all log files sequentially
    info = OrderedDict()
    for file in log_files:
        with open(file, "r") as log_file:
            for line in log_file:
                if "Analysing:" in line:
                    name = line.split()[1]
                    break
            else:
                # If the molecule name isn't found, there's something wrong with the log file
                # To avoid errors, just skip over that file and tell the user.
                print(
                    f"Cannot locate molecule name in {file}\nIs it a valid, QUBEKit-made log file?\n"
                )

        # Create ordered dictionary based on the log file info
        info[name] = _populate_progress_dict(file)

    print("Displaying progress of all analyses in current directory.")
    print(f"Progress key: {COLOURS.green}\u2713{COLOURS.end} = Done;", end=" ")
    print(f"{COLOURS.blue}S{COLOURS.end} = Skipped;", end=" ")
    print(f"{COLOURS.red}E{COLOURS.end} = Error;", end=" ")
    print(f"{COLOURS.orange}R{COLOURS.end} = Running;", end=" ")
    print(f"{COLOURS.purple}~{COLOURS.end} = Queued")

    header_string = "{:15}" + "{:>10}" * 10
    print(
        header_string.format(
            "Name",
            "Param",
            "Pre Opt",
            "QM Opt",
            "Hessian",
            "Mod-Sem",
            "Density",
            "Charges",
            "L-J",
            "Tor Scan",
            "Tor Opt",
        )
    )

    # Sort the info alphabetically
    info = OrderedDict(sorted(info.items(), key=lambda tup: tup[0]))

    # Outer dict contains the names of the molecules.
    for key_out, var_out in info.items():
        print(f"{key_out[:13]:15}", end=" ")

        # Inner dict contains the individual molecules' data.
        for var_in in var_out.values():

            if var_in == "\u2713":
                print(f"{COLOURS.green}{var_in:>9}{COLOURS.end}", end=" ")

            elif var_in == "S":
                print(f"{COLOURS.blue}{var_in:>9}{COLOURS.end}", end=" ")

            elif var_in == "E":
                print(f"{COLOURS.red}{var_in:>9}{COLOURS.end}", end=" ")

            elif var_in == "R":
                print(f"{COLOURS.orange}{var_in:>9}{COLOURS.end}", end=" ")

            elif var_in == "~":
                print(f"{COLOURS.purple}{var_in:>9}{COLOURS.end}", end=" ")

        print("")


def _populate_progress_dict(file_name):
    """
    With a log file open:
        Search for a keyword marking the completion or skipping of a stage;
        If that's not found, look for error messages,
        Otherwise, just return that the stage hasn't finished yet.
    Key:
        tick mark (u2713): Done; S: Skipped; E: Error; ~ (tilde): Neither complete nor errored nor skipped.
    """

    # Indicators in the log file which describe a stage
    search_terms = (
        "PARAMETRISATION",
        "PRE_OPT",
        "QM_OPT",
        "HESSIAN",
        "MOD_SEM",
        "DENSITY",
        "CHARGE",
        "LENNARD",
        "TORSION_S",
        "TORSION_O",
    )

    progress = OrderedDict((term, "~") for term in search_terms)

    restart_log = False

    with open(file_name) as file:
        for line in file:

            if "Continuing log file" in line:
                restart_log = True

            # Look for the specific search terms
            for term in search_terms:
                if term in line:
                    if "SKIP" in line:
                        progress[term] = "S"
                    elif "STARTING" in line:
                        progress[term] = "R"
                    # If its finishing tag is present it is done (u2713 == tick)
                    elif "FINISHING" in line:
                        progress[term] = "\u2713"

            # If an error is found, then the stage after the last successful stage has errored (E)
            if "Exception Logger - ERROR" in line:
                for key, value in progress.items():
                    if value == "R":
                        restart_term = search_terms.index(key)
                        progress[key] = "E"
                        break

    if restart_log:
        for term, stage in progress.items():
            # Find where the program was restarted from
            if stage == "R":
                restart_term = search_terms.index(term)
                break
        else:
            # If no stage is running, find the first stage that hasn't started; the first `~`
            for term, stage in progress.items():
                if stage == "~":
                    restart_term = search_terms.index(term)
                    break

        # Reset anything after the restart term to be `~` even if it was previously completed.
        try:
            for term in search_terms[restart_term + 1 :]:
                progress[term] = "~"
        except UnboundLocalError:
            pass

    return progress


def pretty_print(molecule, to_file=False, finished=True):
    """
    Takes a ligand molecule class object and displays all the class variables in a clean, readable format.

    Print to log: * On exception
                  * On completion
    Print to terminal: * On call
                       * On completion

    Strictly speaking this should probably be a method of ligand class as it explicitly uses ligand's custom
    __str__ method with an extra argument.
    """

    pre_string = (
        f'\n\nOn {"completion" if finished else "exception"}, the ligand objects are:'
    )

    if not to_file:
        pre_string = f"{COLOURS.green}{pre_string}{COLOURS.end}"

    # Print to log file rather than to terminal
    if to_file:
        log_location = os.path.join(molecule.home, "QUBEKit_log.txt")
        with open(log_location, "a+") as log_file:
            log_file.write(f"{pre_string.upper()}\n\n{molecule.__str__()}")

    # Print to terminal
    else:
        print(pre_string)
        # Custom __str__ method; see its documentation for details.
        print(molecule.__str__(trunc=True))
        print("")


def display_molecule_objects(*names):
    """
    prints the requested molecule objects in a nicely formatted way, easy to copy elsewhere.
    To be used via the -display command
    :param names: list of strings where each item is the name of a molecule object such as 'basis' or 'coords'
    """
    try:
        molecule = unpickle()["finalise"]
    except KeyError:
        print(
            "QUBEKit encountered an error during execution; returning the initial molecule objects."
        )
        molecule = unpickle()["parametrise"]

    for name in names:
        result = getattr(molecule, name, None)
        if result is not None:
            print(f"{name}:  {repr(result)}")
        else:
            print(
                f"Invalid molecule object: {name}. Please check the log file for the data you require."
            )
