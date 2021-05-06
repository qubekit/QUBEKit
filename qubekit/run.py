#!/usr/bin/env python3

"""
TODO
    Squash unnecessary arguments into self.molecule. Args such as torsion_options.
        Better handling (or removal?) of torsion_options
    Option to use numbers to skip e.g. -skip 4 5 : skips hessian and mod_seminario steps
    BULK
        Add .sdf as possible bulk_run
        Ideally, input will be a field which takes multiple smiles/pdbs etc.
    Move skip/restart/end/(home?) to Execute rather than ligand.py
        Also needs to be fixed for -bulk
    solvent commands removed:
        Maybe separate solvents into known solvents and IPCM constants?

"""

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime
from functools import partial
from shutil import copy, move
from typing import List

import numpy as np
from tqdm import tqdm

import qubekit
from qubekit.charges import DDECCharges, MBISCharges, extract_extra_sites_onetep
from qubekit.engines import Gaussian, GeometryOptimiser, TorsionDriver, call_qcengine
from qubekit.mod_seminario import ModSeminario
from qubekit.molecules import Ligand
from qubekit.nonbonded.lennard_jones import LennardJones612
from qubekit.nonbonded.virtual_sites import VirtualSites
from qubekit.parametrisation import XML, AnteChamber, OpenFF
from qubekit.torsions import TorsionOptimiser, TorsionScan1D
from qubekit.utils.configs import Configure
from qubekit.utils.constants import COLOURS
from qubekit.utils.datastructures import LocalResource, QCOptions
from qubekit.utils.decorators import exception_logger
from qubekit.utils.display import (
    display_molecule_objects,
    pretty_print,
    pretty_progress,
)
from qubekit.utils.exceptions import (
    GeometryOptimisationError,
    HessianCalculationFailed,
    SpecificationError,
)
from qubekit.utils.file_handling import folder_setup, make_and_change_into
from qubekit.utils.helpers import (
    append_to_log,
    generate_bulk_csv,
    mol_data_from_csv,
    string_to_bool,
    unpickle,
    update_ligand,
)

# To avoid calling flush=True in every print statement.
printf = partial(print, flush=True)


class ArgsAndConfigs:
    """
    This class will be called once for any individual or bulk run.
    As such, this class needs to:
        * Parse all commands from terminal
        * If individual:
            * Initialise molecule
            * Using proper config file (from terminal command or default)
            * Store all configs from file into Molecule object
            * Store any config changes from terminal into Molecule object
            * Call Execute (which handles: working dir, logging, order, etc; see docstring for info)

        * If bulk:
            * For each molecule:
                * Initialise molecule
                * Using proper config file (from terminal command or default)
                * Store all configs from file
                * Store any config changes from terminal
                * Call Execute
    """

    def __init__(self):
        # First make sure the config folder has been made missing for conda and pip
        home = os.path.expanduser("~")
        config_folder = os.path.join(home, "QUBEKit_configs")
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
            printf(f"Making config folder at: {home}")

        self.args = self.parse_commands()

        # If it's a bulk run, handle it separately
        if self.args.bulk_run is not None:
            self.handle_bulk()

        elif self.args.restart is not None:
            # Find the pickled checkpoint file and load it as the molecule
            try:
                self.molecule = update_ligand(self.args.restart, Ligand)
            except KeyError:
                raise KeyError(
                    "This stage was not found in the log file; was the previous stage completed?"
                )
        else:
            # Fresh start; initialise molecule from scratch
            if self.args.smiles:
                self.molecule = Ligand.from_smiles(*self.args.smiles)
            else:
                self.molecule = Ligand.from_file(file_name=self.args.input)

        # Find which config file is being used
        self.molecule.config_file = self.args.config_file

        # Handle configs which are in a file
        file_configs = Configure().load_config(self.molecule.config_file)
        for name, val in file_configs.items():
            setattr(self.molecule, name, val)

        # Although these may be None always, they need to be explicitly set anyway.
        self.molecule.restart = None
        self.molecule.end = None
        self.molecule.skip = None

        # Handle configs which are changed by terminal commands
        for name, val in vars(self.args).items():
            if val is not None:
                setattr(self.molecule, name, val)

        # Now check if we have been supplied a dihedral file and a constraints file
        if self.args.dihedral_file:
            self.molecule.read_scan_order(self.args.dihedral_file)
        if self.args.constraints_file:
            self.molecule.constraints_file = self.args.constaints_file

        # If restarting put the molecule back into the checkpoint file with the new configs
        if self.args.restart is not None:
            self.molecule.pickle(state=self.args.restart)
            self.molecule.home = os.getcwd()

        # Now that all configs are stored correctly: execute.
        Execute(self.molecule)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    @staticmethod
    def parse_commands():
        """
        Parses commands from the terminal using argparse.
        Contains classes for handling actions as well as simple arg parsers for config changes.
        :returns: parsed args
        """

        # Action classes
        class SetupAction(argparse.Action):
            """The setup action class that is called when setup is found in the command line."""

            def __call__(self, pars, namespace, values, option_string=None):
                choice = int(
                    input(
                        "You can now edit config files using QUBEKit, choose an option to continue:\n"
                        "1) Edit a config file\n"
                        "2) Create a new master template\n"
                        "3) Make a normal config file\n"
                        "4) Cancel\n>"
                    )
                )

                if choice == 1:
                    inis = Configure().show_ini()
                    name = input(
                        f"Enter the name or number of the config file to edit\n"
                        f'{"".join(f"{inis.index(ini)}:{ini}    " for ini in inis)}\n>'
                    )
                    # make sure name is right
                    if name in inis:
                        Configure().ini_edit(name)
                    else:
                        Configure().ini_edit(inis[int(name)])

                elif choice == 2:
                    Configure().ini_writer("master_config.ini")
                    Configure().ini_edit("master_config.ini")

                elif choice == 3:
                    name = input("Enter the name of the config file to create\n>")
                    Configure().ini_writer(name)
                    Configure().ini_edit(name)

                else:
                    sys.exit(
                        "Cancelling setup; no changes made. "
                        "If you accidentally entered the wrong key, restart with QUBEKit -setup"
                    )

                sys.exit()

        class CSVAction(argparse.Action):
            """The csv creation class run when the csv option is used."""

            def __call__(self, pars, namespace, values, option_string=None):
                generate_bulk_csv(*values)
                sys.exit()

        class ProgressAction(argparse.Action):
            """Run the pretty progress function to get the progress of all running jobs."""

            def __call__(self, pars, namespace, values, option_string=None):
                pretty_progress()
                sys.exit()

        class DisplayMolAction(argparse.Action):
            """Display the molecule objects requested"""

            def __call__(self, pars, namespace, values, option_string=None):
                display_molecule_objects(*values)
                sys.exit()

        class TorsionTestAction(argparse.Action):
            """
            Using the molecule, test the agreement with QM by doing a torsiondrive and checking the single
            point energies for each rotatable dihedral.
            """

            def __call__(self, pars, namespace, values, option_string=None):
                molecule = update_ligand("finalise", Ligand)
                # If there is a constraints file we should move it
                if molecule.constraints_file is not None:
                    copy(molecule.constraints_file, molecule.constraints_file.name)

                TorsionOptimiser(molecule).torsion_test()

                printf("Torsion testing done!")

                sys.exit()

        intro = (
            "Welcome to QUBEKit! For a list of possible commands, use the help command: -h. "
            "Alternatively, take a look through our github page for commands, recipes and common problems: "
            "https://github.com/qubekit/QUBEKit"
        )
        parser = argparse.ArgumentParser(
            prog="QUBEKit",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=intro,
        )

        # Add all of the command line options in the arg parser
        parser.add_argument(
            "-c",
            "--charge",
            type=int,
            help="Enter the charge of the molecule, default 0.",
        )
        parser.add_argument(
            "-m",
            "--multiplicity",
            type=int,
            help="Enter the multiplicity of the molecule, default 1.",
        )
        parser.add_argument(
            "-threads",
            "--threads",
            type=int,
            help="Number of total threads used in various stages of analysis, especially for engines like "
            "PSI4, Gaussian09, etc. Value is given as an int. Value must be even.",
        )
        parser.add_argument(
            "-memory",
            "--memory",
            type=int,
            help="Amount of total memory used in various stages of analysis, especially for engines like "
            "PSI4, Gaussian09, etc. Value is given as an int, e.g. 6GB is simply 6. Value must be even.",
        )
        parser.add_argument(
            "-n_workers",
            "--n_workers",
            type=int,
            default=1,
            help="The total number of workers which can be used to run QM torsiondrives. Here the total available cores and"
            "memory will be divided between them. Available options are 1,2,4 to prevent waste.",
        )
        parser.add_argument(
            "-ddec",
            "--ddec_version",
            choices=[3, 6],
            type=int,
            help="Enter the ddec version for charge partitioning, does not effect ONETEP partitioning.",
        )
        parser.add_argument(
            "-bonds",
            "--bonds_engine",
            choices=["psi4", "gaussian"],
            help="Choose the QM code to calculate the bonded terms.",
        )
        parser.add_argument(
            "-charges",
            "--charges_engine",
            choices=["onetep", "chargemol", "mbis"],
            default="chargemol",
            help="Choose the method to do the charge partitioning this is tied to the method used to compute the density.",
        )
        parser.add_argument(
            "-convergence",
            "--convergence",
            choices=["GAU", "GAU_TIGHT", "GAU_VERYTIGHT"],
            type=str.upper,
            help="Enter the convergence criteria for the optimisation.",
        )
        parser.add_argument(
            "-param",
            "--parameter_engine",
            choices=["xml", "antechamber", "openff", "none"],
            help="Enter the method of where to get the initial molecule parameters from; "
            "if xml, make sure the xml file has the same name as the input file or smiles input name.",
        )
        parser.add_argument(
            "-pre_opt",
            "--pre_opt_method",
            choices=[
                "mmff94",
                "mmff94s",
                "uff",
                "gfn1xtb",
                "gfn2xtb",
                "gfn0xtb",
                "gaff-2.11",
                "ani1x",
                "ani1ccx",
                "ani2x",
                "openff-1.3.0",
            ],
            help="Enter the optimisation method for pre qm optimisation.",
        )
        parser.add_argument(
            "-config",
            "--config_file",
            choices=Configure().show_ini(),
            help="Enter the name of the configuration file you wish to use for this run from the list "
            "available, defaults to master.",
        )
        parser.add_argument(
            "-theory",
            "--theory",
            help="Enter the name of the qm theory you would like to use.",
        )
        parser.add_argument(
            "-basis", "--basis", help="Enter the basis set you would like to use."
        )
        parser.add_argument(
            "-restart",
            "--restart",
            choices=[
                "parametrise",
                "pre_optimise",
                "qm_optimise",
                "hessian",
                "mod_sem",
                "density",
                "charges",
                "lennard_jones",
                "torsion_scan",
                "torsion_optimise",
            ],
            help="Enter the restart point of a QUBEKit job.",
        )
        parser.add_argument(
            "-end",
            "-end",
            choices=[
                "parametrise",
                "pre_optimise",
                "qm_optimise",
                "hessian",
                "mod_sem",
                "density",
                "charges",
                "lennard_jones",
                "torsion_scan",
                "torsion_optimise",
                "finalise",
            ],
            help="Enter the end point of the QUBEKit job.",
        )
        parser.add_argument(
            "-skip",
            "--skip",
            nargs="+",
            choices=[
                "pre_optimise",
                "qm_optimise",
                "hessian",
                "mod_sem",
                "density",
                "charges",
                "lennard_jones",
                "torsion_scan",
                "torsion_optimise",
                "finalise",
            ],
            help="Option to skip certain stages of the execution.",
        )
        parser.add_argument(
            "-tor_method",
            "--torsion_method",
            choices=["forcebalance", "internal"],
            default="forcebalance",
            help="The method used to optimise the rotatable torsion parameters.",
        )
        parser.add_argument(
            "-tor_test",
            "--torsion_test",
            action=TorsionTestAction,
            help="Enter True if you would like to run a torsion test on the chosen torsions.",
        )
        parser.add_argument(
            "-log",
            "--log",
            type=str,
            help="Enter a name to tag working directories with. Can be any alphanumeric string. "
            "This helps differentiate (by more than just date) different analyses of the "
            "same molecule.",
        )
        parser.add_argument(
            "-vib",
            "--vib_scaling",
            type=float,
            help="Enter the vibrational scaling to be used with the basis set.",
        )
        parser.add_argument(
            "-iters",
            "--iterations",
            type=int,
            help="Max number of iterations for QM scan.",
        )
        parser.add_argument(
            "-constraints",
            "--constraints_file",
            type=str,
            help="The name of the geometric constraints file.",
        )
        parser.add_argument(
            "-dihedrals",
            "--dihedral_file",
            type=str,
            help="The name of the qubekit/tdrive torsion file.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            choices=[True, False],
            type=string_to_bool,
            help="Decide whether the log file should contain all the input/output information",
        )
        parser.add_argument(
            "-symmetry",
            "--enable_symmetry",
            choices=[True, False],
            type=string_to_bool,
            help="Enable or disable the use of symmetrisation for bond, angle, charge, and "
            "Lennard-Jones parameters",
        )
        parser.add_argument(
            "-sites",
            "--enable_virtual_sites",
            choices=[True, False],
            type=string_to_bool,
            help="Enable or disable the use of virtual sites in the charge fitting.",
        )
        parser.add_argument(
            "-site_err",
            "--v_site_error_factor",
            type=float,
            help="Maximum error factor from adding a site that means the site will be kept",
        )

        # Add mutually exclusive groups to stop certain combinations of options,
        # e.g. setup should not be run with csv command
        groups = parser.add_mutually_exclusive_group()
        groups.add_argument(
            "-setup",
            "--setup_config",
            nargs="?",
            const=True,
            help="Setup a new configuration or edit an existing one.",
            action=SetupAction,
        )
        groups.add_argument(
            "-sm",
            "--smiles",
            nargs="+",
            help="Enter the smiles string of a molecule as a starting point.",
        )
        groups.add_argument(
            "-bulk",
            "--bulk_run",
            help="Enter the name of the csv file to run as bulk, bulk will use smiles unless it finds "
            "a molecule file with the same name.",
        )
        groups.add_argument(
            "-csv",
            "--csv_filename",
            action=CSVAction,
            nargs="*",
            help="Enter the name of the csv file you would like to create for bulk runs. "
            "Optionally, you may also add the maximum number of molecules per file.",
        )
        groups.add_argument(
            "-i", "--input", help="Enter the molecule input pdb file (only pdb so far!)"
        )
        groups.add_argument(
            "-version", "--version", action="version", version=qubekit.__version__
        )
        groups.add_argument(
            "-progress",
            "--progress",
            nargs="?",
            const=True,
            action=ProgressAction,
            help="Get the current progress of a QUBEKit single or bulk job.",
        )
        groups.add_argument(
            "-display",
            "--display",
            type=str,
            nargs="+",
            action=DisplayMolAction,
            help="Get the molecule object with this name in the cwd",
        )

        # Ensures help is shown (rather than an error) if no arguments are provided.
        return parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    def handle_bulk(self):
        """
        Getting and setting configs for bulk runs is a little different, requiring this method.
        The configs are taken from the .csv, then the .ini, then the terminal.
        This is repeated for each molecule in the bulk run, then Execute is called.

        Configs cannot be changed between molecule analyses as config data is
        only loaded once at the start; -restart is required for that.
        """

        csv_file = self.args.bulk_run
        # mol_data_from_csv handles defaults if no argument is given
        bulk_data = mol_data_from_csv(csv_file)

        names = list(bulk_data)

        home = os.getcwd()

        for name in names:
            printf(f"Analysing: {name}\n")

            # Get pdb from smiles or name if no smiles is given
            if bulk_data[name]["smiles"] is not None:
                smiles_string = bulk_data[name]["smiles"]
                printf(f"smile string: {smiles_string}\n")
                self.molecule = Ligand.from_smiles(smiles_string, name)

            else:
                # Initialise molecule, ready to add configs to it
                self.molecule = Ligand.from_file(f"{name}.pdb")

            # Read each row in bulk data and set it to the molecule object
            for key, val in bulk_data[name].items():
                setattr(self.molecule, key, val)

            self.molecule.skip = None

            # Using the config file from the .csv, gather the .ini file configs
            file_configs = Configure().load_config(self.molecule.config_file)
            for key, val in file_configs.items():
                setattr(self.molecule, key, val)

            # Handle configs which are changed by terminal commands
            for key, val in vars(self.args).items():
                if val is not None:
                    setattr(self.molecule, key, val)

            # Now that all configs are stored correctly: execute.
            Execute(self.molecule)

            os.chdir(home)

        sys.exit(
            f"{COLOURS.green}Bulk analysis complete.{COLOURS.end}\n"
            "Use QUBEKit -progress to view the completion progress of your molecules"
        )


class Execute:
    """
    This class will be called for each INDIVIDUAL run and for each analysis in a BULK run.
    ArgsAndConfigs will handle bulk runs, calling this class as and when necessary.

    As such, this class needs to handle for each run:
        * Create and / or move into working dir
        * Create log file (Skipped if restarting)
        * Write to log file what is happening at the start of execution
            (starting or continuing; what are the configs; etc)
        * Execute any stages required, in the correct order according to self.order
        * Store info to log file as the program runs
            (which stages are being called; any key info / timings regarding progress; errors; etc)
        * Return results, both in individual stage directories and at the end
    """

    def __init__(self, molecule: Ligand):

        # At this point, molecule should contain all of the config options from
        # the defaults, config file and terminal commands. These all come from the ArgsAndConfigs class.
        self.molecule = molecule

        self.start_up_msg = (
            "If QUBEKit ever breaks or you would like to view timings and loads of other info, "
            "view the log file.\nOur documentation (README.md) "
            "also contains help on handling the various commands for QUBEKit.\n"
        )

        self.order = OrderedDict(
            [
                ("parametrise", self.parametrise),
                ("pre_optimise", self.pre_optimise),
                ("qm_optimise", self.qm_optimise),
                ("hessian", self.hessian),
                ("mod_sem", self.mod_sem),
                ("charges", self.charges),
                ("lennard_jones", self.lennard_jones),
                ("torsion_scan", self.torsion_scan),
                ("torsion_optimise", self.torsion_optimise),
                ("finalise", self.finalise),
            ]
        )

        # Keep this for reference (used for numbering folders correctly)
        self.immutable_order = tuple(self.order)

        self.engine_dict = {"g09": Gaussian, "g16": Gaussian}

        printf(self.start_up_msg)

        # If restart is None, then the analysis has not been started previously
        self.create_log() if self.molecule.restart is None else self.continue_log()

        self.redefine_order()
        # write the molecule file if not already done
        self.molecule.to_file(file_name=f"{molecule.name}.pdb")
        self.run()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def redefine_order(self):
        """
        If any order changes are required (restarting, new end point, skipping stages), it is done here.
        Creates a new self.order based on self.molecule's configs.
        """

        start = (
            self.molecule.restart
            if self.molecule.restart is not None
            else "parametrise"
        )
        end = self.molecule.end if self.molecule.end is not None else "finalise"
        skip = self.molecule.skip if self.molecule.skip is not None else []

        # Create list of all keys
        stages = list(self.order)

        # Cut out the keys before the start_point and after the end_point
        # Add finalise back in if it's removed (finalise should always be called).
        stages = stages[stages.index(start) : stages.index(end) + 1] + ["finalise"]

        # Redefine self.order to only contain the key, val pairs from stages
        self.order = OrderedDict(
            pair for pair in self.order.items() if pair[0] in set(stages)
        )

        for pair in self.order.items():
            self.order[pair[0]] = self.skip if pair[0] in skip else pair[1]

    def create_log(self):
        """
        Creates the working directory for the job as well as the log file.
        This log file is then extended when:
            - helpers.append_to_log() is called;
            - helpers.pretty_print() is called with to_file set to True;
            - decorators.exception_logger() wraps a function / method which throws an exception.
                (This captures almost all exceptions, rethrows important ones)

        This method also makes backups if the working directory already exists.
        """

        date = datetime.now().strftime("%Y_%m_%d")

        # Define name of working directory.
        # This is formatted as 'QUBEKit_molecule name_yyyy_mm_dd_log_string'.

        dir_name = f"QUBEKit_{self.molecule.name}_{date}_{self.molecule.log}"

        # If you can't make a dir because it exists, back it up.
        try:
            os.mkdir(dir_name)

        except FileExistsError:
            # Try make a backup folder
            try:
                os.mkdir("QUBEKit_backups")
                printf("Making backup folder: QUBEKit_backups")
            except FileExistsError:
                # Backup folder already made
                pass
            # Keep increasing backup number until that particular number does not exist
            finally:
                count = 1
                while os.path.exists(
                    f"QUBEKit_backups/{dir_name}_{str(count).zfill(3)}"
                ):
                    count += 1
                    if count >= 100:
                        raise RuntimeError("Cannot create more than 100 backups.")

                # Then, make that backup and make a new working directory
                move(dir_name, f"QUBEKit_backups/{dir_name}_{str(count).zfill(3)}")
                printf(f"Moving directory: {dir_name} to backup folder")
                os.mkdir(dir_name)

        # Finally, having made any necessary backups, move files and change to working dir.
        finally:
            os.chdir(dir_name)

            # Set a home directory
            self.molecule.home = os.getcwd()

        # Find external files
        copy_files = [f"{self.molecule.name}.xml", f"{self.molecule.name}.pdb"]
        for file in copy_files:
            try:
                copy(f"../{file}", file)
            except (FileNotFoundError, TypeError):
                pass

        with open("QUBEKit_log.txt", "w+") as log_file:
            log_file.write(
                f"Beginning log file; the time is: {datetime.now()}\n\n\n"
                f"Your current QUBEKit version is: {qubekit.__version__}\n\n\n"
            )

        self.log_configs()

    def continue_log(self):
        """
        In the event of restarting an analysis, find and append to the existing log file
        rather than creating a new one.
        """

        with open("QUBEKit_log.txt", "a+") as log_file:
            log_file.write(
                f"\n\nContinuing log file from previous execution; the time is: {datetime.now()}\n\n\n"
            )

        self.log_configs()

    def log_configs(self):
        """Writes the runtime and file-based defaults to a log file."""

        with open("QUBEKit_log.txt", "a+") as log_file:

            log_file.write(f"Analysing: {self.molecule.name}\n\n")

            if self.molecule.verbose:
                log_file.write(
                    "The runtime defaults and config options are as follows:\n\n"
                )
                for key, val in self.molecule.__dict__.items():
                    if val is not None:
                        log_file.write(f"{key}: {val}\n")
                log_file.write("\n")
            else:
                log_file.write(
                    "Ligand state not logged; verbose argument set to false\n\n"
                )

    @exception_logger
    def run(self, torsion_options=None):
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order
            (according to self.order).
        Exceptions are added to log (if raised) using the decorator.
        """

        if "parametrise" in self.order:
            if torsion_options is not None:
                torsion_options = torsion_options.split(",")
                self.molecule = self.store_torsions(self.molecule, torsion_options)
            self.molecule.pickle(state="parametrise")

        stage_dict = {
            # Stage: [Start message, End message]
            "parametrise": [
                f"Parametrising molecule with {self.molecule.parameter_engine}",
                "Molecule parametrised",
            ],
            "pre_optimise": [
                f"Partially optimising with {self.molecule.pre_opt_method}",
                "Partial optimisation complete",
            ],
            "qm_optimise": [
                f"Optimising molecule with QM with {self.molecule.basis} / {self.molecule.theory} using "
                f"{self.molecule.bonds_engine}",
                "Molecule optimisation complete",
            ],
            "hessian": [
                "Calculating Hessian matrix",
                "Hessian matrix calculated and confirmed to be symmetric",
            ],
            "mod_sem": [
                "Calculating bonds and angles with modified Seminario method",
                "Bond and angle parameters calculated",
            ],
            "charges": [
                f"{self.molecule.charges_engine} calculating charges",
                "Charges calculated",
            ],
            "lennard_jones": [
                "Performing Lennard-Jones calculation",
                "Lennard-Jones parameters calculated",
            ],
            "torsion_scan": [
                f"Performing QM-constrained optimisation with Torsiondrive and {self.molecule.bonds_engine}",
                "Torsiondrive finished and QM results saved",
            ],
            "torsion_optimise": [
                "Performing torsion optimisation",
                "Torsion optimisation complete",
            ],
            "finalise": ["Finalising analysis", "Molecule analysis complete!"],
            "pause": ["Pausing analysis", "Analysis paused!"],
            "skip": ["Skipping section", "Section skipped"],
        }

        # Do the first stage in the order to get the next_key for the following loop
        key = list(self.order)[0]
        next_key = self.stage_wrapper(key, *stage_dict[key], torsion_options)

        # Cannot use for loop as we mutate the dictionary during the loop
        while True:
            if next_key is None:
                break
            next_key = self.stage_wrapper(next_key, *stage_dict[next_key])

            if next_key == "pause":
                self.pause()
                break

    def stage_wrapper(
        self, start_key, begin_log_msg="", fin_log_msg="", torsion_options=None
    ):
        """
        Firstly, check if the stage start_key is in self.order; this tells you if the stage should be called or not.
        If it isn't in self.order:
            - Do nothing
        If it is:
            - Unpickle the ligand object at the start_key stage
            - Write to the log that something's about to be done (if specified)
            - Make (if not restarting) and / or move into the working directory for that stage
            - Do the thing
            - Move back out of the working directory for that stage
            - Write to the log that something's been done (if specified)
            - Pickle the ligand object again with the next_key marker as its stage
        """

        mol = unpickle()[start_key]

        # Set the state for logging any exceptions should they arise
        mol.state = start_key

        # if we have a torsion options dictionary pass it to the molecule
        if torsion_options is not None:
            mol = self.store_torsions(mol, torsion_options)

        # Handle skipping of a stage
        skipping = False
        if self.order[start_key] == self.skip:
            printf(f"{COLOURS.blue}Skipping stage: {start_key}{COLOURS.end}")
            append_to_log(mol.home, f"skipping stage: {start_key}", major=True)
            skipping = True
        else:
            if begin_log_msg:
                printf(f"{begin_log_msg}...", end=" ")

        home = os.getcwd()

        folder_name = (
            f"{str(self.immutable_order.index(start_key) + 1).zfill(2)}_{start_key}"
        )

        make_and_change_into(folder_name)

        mol = self.order[start_key](mol)
        self.order.pop(start_key, None)
        os.chdir(home)

        # Begin looping through self.order, but return after the first iteration.
        for key in self.order:
            next_key = key
            if fin_log_msg and not skipping:
                printf(f"{COLOURS.green}{fin_log_msg}{COLOURS.end}")

            mol.pickle(state=next_key)
            return next_key

    @staticmethod
    def parametrise(molecule: Ligand, verbose: bool = True) -> Ligand:
        """Perform initial molecule parametrisation using OpenFF, Antechamber or XML."""
        if verbose:
            append_to_log(molecule.home, "Starting parametrisation", major=True)

        # Parametrisation options:
        param_dict = {"antechamber": AnteChamber(), "xml": XML(), "openff": OpenFF()}

        # If we are using xml we have to move it to QUBEKit working dir
        input_files = []
        if molecule.parameter_engine == "xml":
            xml_name = f"{molecule.name}.xml"
            input_files.append(xml_name)
            if xml_name not in os.listdir("."):
                try:
                    copy(
                        os.path.join(molecule.home, f"{molecule.name}.xml"),
                        f"{molecule.name}.xml",
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "You need to supply an xml file if you wish to use xml-based parametrisation; "
                        "put this file in the location you are running QUBEKit from. "
                        "Alternatively, use a different parametrisation method such as: "
                        "-param openff"
                    )

        # Perform the parametrisation
        param_method = param_dict[molecule.parameter_engine]
        param_mol = param_method.run(molecule=molecule, input_files=input_files)

        if verbose:
            append_to_log(
                molecule.home,
                f"Finishing parametrisation of molecule with {molecule.parameter_engine}",
                major=True,
            )

        return param_mol

    @staticmethod
    def pre_optimise(molecule: Ligand) -> Ligand:
        """
        Do a pre optimisation of the molecule using the specified program.

        options
        ---------
        "mmff94", "uff", "mmff94s", "gfn1xtb", "gfn2xtb", "fgn0xtb", "gaff-2.11", "ani1x", "ani1ccx", "ani2x", "openff-1.3.0

        """
        from copy import deepcopy
        from multiprocessing import Pool

        # TODO drop all of this once we change configs
        # now we want to build the optimiser from the inputs
        method = molecule.pre_opt_method.lower()
        if method in ["mmff94", "mmff94s", "uff"]:
            program = "rdkit"
            basis = None
        elif method in ["gfn1xtb", "gfn2xtb", "gfn0xtb"]:
            program = "xtb"
            basis = None
        elif method in ["ani1x", "ani1ccx", "ani2x"]:
            program = "torchani"
            basis = None
        elif method == "gaff-2.11":
            program = "openmm"
            basis = "antechamber"
        elif method == "openff-1.3.0":
            program = "openmm"
            basis = "smirnoff"
        else:
            raise SpecificationError(
                f"The pre optimisation method {method} is not supported, please choose from "
                f"mmff94, mmff94s, uff, gfn1xtb, gfn2xtb, gfn0xtb, gaff-2.11, ani1x, ani1ccx, ani2x, openff-1.3.0"
            )

        append_to_log(
            molecule.home,
            f"Starting pre_optimisation with program: {program} basis: {basis} method: {method}",
            major=True,
        )
        qc_spec = QCOptions(program=program, method=method, basis=basis)
        local_ops = LocalResource(cores=1, memory=1)
        g_opt = GeometryOptimiser(
            convergence="GAU",
            maxiter=molecule.iterations,
        )

        # get some extra conformations
        # total of 10 including input, so 9 new ones
        geometries = molecule.generate_conformers(n_conformers=10)
        molecule.to_multiconformer_file(
            file_name="starting_coords.xyz", positions=geometries
        )
        opt_list = []
        with Pool(processes=molecule.threads) as pool:
            for confomer in geometries:
                opt_mol = deepcopy(molecule)
                opt_mol.coordinates = confomer
                opt_list.append(pool.apply_async(g_opt.optimise, (opt_mol, True, True)))

            results = []
            for result in tqdm(
                opt_list,
                desc=f"Optimising conformers with {molecule.pre_opt_method}",
                total=len(opt_list),
                ncols=80,
            ):
                # errors are auto raised from the class so catch the result, and write to file
                result_mol, opt_result = result.get()
                if opt_result.success:
                    # save the final energy and molecule
                    results.append((result_mol, opt_result.energies[-1]))
                else:
                    # save the molecule and final energy from the last step if it fails
                    results.append((result_mol, opt_result.input_data["energies"][-1]))

        # sort the results
        results.sort(key=lambda x: x[1])
        final_geometries = [re[0].coordinates for re in results]
        # write all conformers out
        molecule.to_multiconformer_file(
            file_name="mutli_opt.xyz", positions=final_geometries
        )
        # save the lowest energy conformer
        final_mol = results[0][0]
        final_mol.conformers = final_geometries

        append_to_log(
            molecule.home,
            f"Finishing pre_optimisation of the molecule with {molecule.pre_opt_method}",
            major=True,
        )
        return final_mol

    @staticmethod
    def qm_optimise(molecule: Ligand) -> Ligand:
        """
        Optimise the molecule using qm via qcengine.

        Note:
            This method will work through each conformer provided trying to optimise each one to gau_tight, if it fails
            in 50 steps the coords are randomly bumped and we try for another 50 steps, if this still fails we move to
            the next set of starting coords and repeat.
        """
        from copy import deepcopy

        append_to_log(
            molecule.home,
            f"Starting qm_optimisation with program: {molecule.bonds_engine} basis: {molecule.basis} method: {molecule.theory}",
            major=True,
        )
        # TODO get this logic contained in one stage class with the pre_opt stage as well
        g_opt = GeometryOptimiser(
            program=molecule.bonds_engine,
            method=molecule.theory,
            basis=molecule.basis,
            convergence="GAU_TIGHT",
            # lower the maxiter but try multiple coords
            maxiter=50,
        )
        geometries: List[np.ndarray] = molecule.conformers

        opt_mol = deepcopy(molecule)

        for i, conformer in enumerate(
            tqdm(
                geometries, desc="Optimising conformer", total=len(geometries), ncols=80
            )
        ):
            with folder_setup(folder_name=f"conformer_{i}"):
                # set the coords
                opt_mol.coordinates = conformer
                # errors are auto raised from the class so catch the result, and write to file
                qm_result, result = g_opt.optimise(
                    molecule=opt_mol, allow_fail=True, return_result=True
                )
                if result.success:
                    append_to_log(molecule.home, "Conformer optimised to GAU TIGHT")
                    break
                else:
                    append_to_log(molecule.home, "Bumping coordinates and restarting")
                    # grab last coords and bump
                    coords = qm_result.coordinates + np.random.choice(
                        a=[0, 0.01], size=(qm_result.n_atoms, 3)
                    )
                    opt_mol.coordinates = coords
                    bump_mol, bump_result = g_opt.optimise(
                        molecule=opt_mol, allow_fail=True, return_result=True
                    )
                    if bump_result.success:
                        qm_result = bump_mol
                        append_to_log(molecule.home, "Conformer optimised to GAU TIGHT")
                        break

        else:
            raise GeometryOptimisationError(
                "No molecule conformer could be optimised to GAU TIGHT"
            )

        append_to_log(molecule.home, f"QM optimisation finished", major=True)

        return qm_result

    @staticmethod
    def hessian(molecule: Ligand) -> Ligand:
        """Using the assigned bonds engine, calculate the Hessian matrix and store in atomic units."""
        from qubekit.utils.helpers import check_symmetry

        append_to_log(molecule.home, "Starting hessian calculation", major=True)
        # build the QM engine
        # qm_engine = QCEngine(
        #     program=molecule.bonds_engine,
        #     method=molecule.theory,
        #     basis=molecule.basis,
        #     driver="hessian",
        # )
        # result = qm_engine.call_qcengine(molecule=molecule)
        #
        # if not result.success:
        #     raise HessianCalculationFailed(
        #         "The hessian was not calculated check the log file."
        #     )
        #
        # np.savetxt("hessian.txt", result.return_result)
        #
        # molecule.hessian = result.return_result
        # check_symmetry(molecule.hessian)
        #
        # append_to_log(
        #     molecule.home,
        #     f"Finishing Hessian calculation using {molecule.bonds_engine}",
        #     major=True,
        # )

        return molecule

    @staticmethod
    def mod_sem(molecule: Ligand) -> Ligand:
        """Modified Seminario for bonds and angles."""

        append_to_log(molecule.home, "Starting mod_Seminario method", major=True)

        mod_sem = ModSeminario(
            vibrational_scaling=molecule.vib_scaling,
            symmetrise_parameters=molecule.enable_symmetry,
        )
        mod_molecule = mod_sem.run(molecule=molecule)

        append_to_log(molecule.home, "Finishing Mod_Seminario method", major=True)

        return mod_molecule

    def charges(self, molecule: Ligand) -> Ligand:
        """Perform an AIM analysis using MBIS/Chargemol or ONETEP."""

        append_to_log(
            molecule.home,
            f"Starting charge partitioning using {molecule.charges_engine}",
            major=True,
        )
        common_settings = dict(
            basis=molecule.basis,
            method=molecule.theory,
            memory=molecule.memory,
            cores=molecule.threads,
            apply_symmetry=molecule.enable_symmetry,
        )
        if molecule.charges_engine == "chargemol":
            density_engine = DDECCharges(**common_settings)
            density_engine.ddec_version = molecule.ddec_version
            density_engine.solvent_settings.epsilon = molecule.dielectric

        elif molecule.charges_engine == "mbis":
            density_engine = MBISCharges(**common_settings)

        elif molecule.charges_engine == "onetep":
            # check for results files first
            if os.path.exists("iter_1/ddec.onetep") or os.path.exists("ddec.onetep"):
                from qubekit.charges import ExtractChargeData

                # load the charge info
                append_to_log(
                    molecule.home,
                    "Extracting Charge information from ONETEP result",
                    major=True,
                )
                ExtractChargeData.extract_charge_data_onetep(
                    molecule=molecule, dir_path=""
                )
                if molecule.enable_virtual_sites:
                    append_to_log(
                        molecule.home, "Starting virtual sites calculation", major=True
                    )
                    # grab onetep v-sites
                    extract_extra_sites_onetep(molecule)
            else:
                molecule.to_file(file_name=f"{molecule.name}.xyz")
                # If using ONETEP, stop after this step
                # TODO add better ONETEP support via ASE
                append_to_log(
                    molecule.home, "Density analysis file made for ONETEP", major=True
                )

                # Edit the order to end here
                self.order = OrderedDict(
                    [
                        ("charges", self.charges),
                        ("lennard_jones", self.skip),
                        ("torsion_scan", self.skip),
                        ("pause", self.pause),
                        ("finalise", self.finalise),
                    ]
                )
                return molecule
        else:
            raise NotImplementedError

        molecule = density_engine.run(molecule=molecule)

        append_to_log(
            molecule.home,
            f"Finishing charge partitioning with {molecule.charges_engine}",
            major=True,
        )
        if molecule.enable_virtual_sites:
            append_to_log(
                molecule.home, "Starting virtual sites calculation", major=True
            )
            vs = VirtualSites(
                enable_symmetry=molecule.enable_symmetry,
                site_error_factor=molecule.v_site_error_factor,
            )
            vs.run(molecule=molecule)

            append_to_log(
                molecule.home, "Finishing virtual sites calculation", major=True
            )

        return molecule

    @staticmethod
    def lennard_jones(molecule):
        """Calculate Lennard-Jones parameters, and extract virtual sites."""

        append_to_log(
            molecule.home, "Starting Lennard-Jones parameter calculation", major=True
        )
        lj = LennardJones612()
        lj.extract_rfrees()
        lj.run(molecule=molecule)

        append_to_log(
            molecule.home, "Finishing Lennard-Jones parameter calculation", major=True
        )

        return molecule

    @staticmethod
    def torsion_scan(molecule: Ligand) -> Ligand:
        """Perform torsion scan."""

        append_to_log(molecule.home, "Starting torsion_scans", major=True)

        # build the torsiondriver engine
        tdriver = TorsionDriver(
            program=molecule.bonds_engine,
            basis=molecule.basis,
            method=molecule.theory,
            cores=molecule.threads,
            memory=molecule.memory,
            n_workers=molecule.n_workers,
        )
        tor_scan = TorsionScan1D(torsion_driver=tdriver)
        result_mol = tor_scan.run(molecule=molecule)

        append_to_log(molecule.home, "Finishing torsion_scans", major=True)

        return result_mol

    @staticmethod
    def torsion_optimise(molecule: Ligand) -> Ligand:
        """Perform torsion optimisation if there is QM reference data else skip."""

        if molecule.qm_scans is None or molecule.qm_scans == []:
            append_to_log(
                molecule.home,
                "No QM reference data found, no need for torsion_optimisation stage.",
                major=True,
            )
            return molecule

        else:
            append_to_log(molecule.home, "Starting torsion_optimisations", major=True)
            if molecule.torsion_method == "internal":
                fit_molecule = molecule
                TorsionOptimiser(molecule).run()
            elif molecule.torsion_method == "forcebalance":
                from qubekit.torsions import ForceBalanceFitting

                # TODO expose any extra configs?
                fb_fit = ForceBalanceFitting()
                fit_molecule = fb_fit.run(molecule=molecule)
                append_to_log(
                    molecule.home, "Finishing torsion_optimisations", major=True
                )

            else:
                raise RuntimeError(
                    f"The method {molecule.torsion_method} is not supported please chose from internal or forcebalance."
                )

            return fit_molecule

    @staticmethod
    def finalise(molecule: Ligand) -> Ligand:
        """
        Make the xml and pdb file;
        print the ligand object to terminal (in abbreviated form) and to the log file (unabbreviated).
        """
        # Ensure the net charge is an integer value and adds up to molecule.charge
        molecule.fix_net_charge()

        molecule.to_file(file_name=f"{molecule.name}.pdb")
        molecule.write_parameters(file_name=f"{molecule.name}.xml")

        if molecule.verbose:
            pretty_print(molecule, to_file=True)
            pretty_print(molecule)
        else:
            printf(f"{COLOURS.green}Analysis complete!{COLOURS.end}")

        return molecule

    @staticmethod
    def skip(molecule):
        """A blank method that does nothing to that stage but adds the pickle points to not break the flow."""

        return molecule

    @staticmethod
    def pause():
        """
        Pause the analysis when using ONETEP so we can come back into the workflow
        without breaking the pickling process.
        """

        printf(
            "QUBEKit stopping at ONETEP step!\n To continue please move the ddec.onetep file and xyz file to the "
            "density folder and use QUBEKit -restart charges."
        )

        return

    @staticmethod
    def store_torsions(molecule, torsions_list=None):
        """
        Take the molecule object and the list of torsions and convert them to rotatable centres.
        Then, put them in the scan order object.
        """

        if torsions_list is not None:

            scan_order = []
            for torsion in torsions_list:
                tor = tuple(atom for atom in torsion.split("-"))
                # convert the string names to the index
                core = (
                    molecule.get_atom_with_name(tor[1]).atom_index,
                    molecule.get_atom_with_name(tor[2]).atom_index,
                )

                if core in molecule.rotatable:
                    scan_order.append(core)
                elif reversed(core) in molecule.rotatable:
                    scan_order.append(reversed(core))

            molecule.scan_order = scan_order

        return molecule


def main():
    """Putting class call inside a function squashes the __repr__ call"""

    ArgsAndConfigs()


if __name__ == "__main__":
    # For running with debugger, normal entry point is defined in setup.py
    main()
