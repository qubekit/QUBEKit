#!/usr/bin/env python3

"""
TODO
    Squash unnecessary arguments into self.molecule. Args such as torsion_options.
        Better handling (or removal?) of torsion_options
    Option to use numbers to skip e.g. -skip 4 5 : skips hessian and mod_seminario steps
    BULK
        Add .sdf as possible bulk_run, not just .csv
        Bulk torsion options need to be made easier to use
"""

import argparse
import os
import subprocess as sp
import sys
from collections import OrderedDict
from datetime import datetime
from functools import partial
from shutil import copy, move

import numpy as np

from QUBEKit.dihedrals import TorsionOptimiser, TorsionScan
from QUBEKit.engines import PSI4, Chargemol, Gaussian, QCEngine, RDKit
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.ligand import Ligand
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.parametrisation import XML, AnteChamber, OpenFF, Parametrisation
from QUBEKit.utils import constants
from QUBEKit.utils.configs import Configure
from QUBEKit.utils.constants import COLOURS
from QUBEKit.utils.decorators import exception_logger
from QUBEKit.utils.display import (
    display_molecule_objects,
    pretty_print,
    pretty_progress,
)
from QUBEKit.utils.exceptions import HessianCalculationFailed, OptimisationFailed
from QUBEKit.utils.file_handling import (
    ExtractChargeData,
    extract_extra_sites_onetep,
    make_and_change_into,
)
from QUBEKit.utils.helpers import (
    append_to_log,
    fix_net_charge,
    generate_bulk_csv,
    mol_data_from_csv,
    string_to_bool,
    unpickle,
    update_ligand,
)
from QUBEKit.virtual_sites import VirtualSites
import QUBEKit

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

        # If we are doing the torsion test add the attribute to the molecule so we can catch it in execute
        if self.args.torsion_test:
            self.args.restart = "finalise"

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
                self.molecule = Ligand(self.args.input)

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

        # Now we need to remove torsion_test as it is passed from the command line
        # is False to check it's not True nor None
        if self.args.torsion_test is False:
            delattr(self.molecule, "torsion_test")

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

        class TorsionMakerAction(argparse.Action):
            """Help the user make a torsion scan file."""

            def __call__(self, pars, namespace, values, option_string=None):
                # load in the ligand
                mol = Ligand(values)

                # Prompt the user for the scan order
                scanner = TorsionScan(mol)
                scanner.find_scan_order()

                # Write out the scan file
                with open(f"{mol.name}.dihedrals", "w+") as qube:
                    qube.write(
                        "# dihedral definition by atom indices starting from 0\n#  i      j      k      l\n"
                    )
                    for scan in mol.scan_order:
                        scan_di = mol.dihedrals[scan][0]
                        qube.write(
                            f"  {scan_di[0]:2}     {scan_di[1]:2}     {scan_di[2]:2}     {scan_di[3]:2}\n"
                        )
                printf(f"{mol.name}.dihedrals made.")

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
            help="Number of threads used in various stages of analysis, especially for engines like "
            "PSI4, Gaussian09, etc. Value is given as an int.",
        )
        parser.add_argument(
            "-memory",
            "--memory",
            type=int,
            help="Amount of memory used in various stages of analysis, especially for engines like "
            "PSI4, Gaussian09, etc. Value is given as an int, e.g. 6GB is simply 6.",
        )
        parser.add_argument(
            "-ddec",
            "--ddec_version",
            choices=[3, 6],
            type=int,
            help="Enter the ddec version for charge partitioning, does not effect ONETEP partitioning.",
        )
        parser.add_argument(
            "-geo",
            "--geometric",
            choices=[True, False],
            type=string_to_bool,
            help="Turn on geometric to use this during the qm optimisations, recommended.",
        )
        parser.add_argument(
            "-bonds",
            "--bonds_engine",
            choices=["psi4", "g09", "g16"],
            help="Choose the QM code to calculate the bonded terms.",
        )
        parser.add_argument(
            "-charges",
            "--charges_engine",
            choices=["onetep", "chargemol"],
            help="Choose the method to do the charge partitioning.",
        )
        parser.add_argument(
            "-density",
            "--density_engine",
            choices=["onetep", "g09", "g16", "psi4"],
            help="Enter the name of the QM code to calculate the electron density of the molecule.",
        )
        parser.add_argument(
            "-solvent",
            "--solvent",
            choices=[True, False],
            type=string_to_bool,
            help="Enter whether or not you would like to use a solvent.",
        )
        # Maybe separate into known solvents and IPCM constants?
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
            help="Enter the method of where we should get the initial molecule parameters from, "
            "if xml make sure the xml has the same name as the pdb file.",
        )
        parser.add_argument(
            "-mm",
            "--mm_opt_method",
            choices=["openmm", "rdkit_mff", "rdkit_uff", "none"],
            help="Enter the mm optimisation method for pre qm optimisation.",
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
                "mm_optimise",
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
                "mm_optimise",
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
            "-progress",
            "--progress",
            nargs="?",
            const=True,
            help="Get the current progress of a QUBEKit single or bulk job.",
            action=ProgressAction,
        )
        parser.add_argument(
            "-skip",
            "--skip",
            nargs="+",
            choices=[
                "mm_optimise",
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
            "-tor_test",
            "--torsion_test",
            action="store_true",
            help="Enter True if you would like to run a torsion test on the chosen torsions.",
        )
        parser.add_argument(
            "-tor_make",
            "--torsion_maker",
            action=TorsionMakerAction,
            help="Allow QUBEKit to help you make a torsion input file for the given molecule",
        )
        parser.add_argument(
            "-log",
            "--log",
            type=str,
            help="Enter a name to tag working directories with. Can be any alphanumeric string."
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
            "-display",
            "--display",
            type=str,
            nargs="+",
            action=DisplayMolAction,
            help="Get the molecule object with this name in the cwd",
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
        groups.add_argument("-version", "--version", action="version", version="2.6.3")

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
                self.molecule = Ligand(smiles_string, name)

            else:
                # Initialise molecule, ready to add configs to it
                self.molecule = Ligand(f"{name}.pdb")

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

    def __init__(self, molecule):

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
                ("mm_optimise", self.mm_optimise),
                ("qm_optimise", self.qm_optimise),
                ("hessian", self.hessian),
                ("mod_sem", self.mod_sem),
                ("density", self.density),
                ("charges", self.charges),
                ("lennard_jones", self.lennard_jones),
                ("torsion_scan", self.torsion_scan),
                ("torsion_optimise", self.torsion_optimise),
                ("finalise", self.finalise),
                ("torsion_test", self.torsion_test),
            ]
        )

        # Keep this for reference (used for numbering folders correctly)
        self.immutable_order = tuple(self.order)

        self.engine_dict = {"psi4": PSI4, "g09": Gaussian, "g16": Gaussian}

        printf(self.start_up_msg)

        # If restart is None, then the analysis has not been started previously
        self.create_log() if self.molecule.restart is None else self.continue_log()

        self.redefine_order()

        self.run()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def redefine_order(self):
        """
        If any order changes are required (restarting, new end point, skipping stages), it is done here.
        Creates a new self.order based on self.molecule's configs.
        """

        # If we are doing a torsion_test we have to redo the order as follows:
        # 1 Skip the finalise step to create a pickled ligand at the torsion_test stage
        # 2 Do the torsion_test and delete the torsion_test attribute
        # 3 Do finalise again to save the ligand with the correct attributes
        if getattr(self.molecule, "torsion_test", None) not in [None, False]:
            self.order = OrderedDict(
                [
                    ("finalise", self.skip),
                    ("torsion_test", self.torsion_test),
                    ("finalise", self.skip),
                ]
            )

        else:
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
                f"Your current QUBEKit version is: {QUBEKit.__version__}\n\n\n"
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
            "mm_optimise": [
                "Partially optimising with MM",
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
            "density": [
                f"Performing density calculation with {self.molecule.density_engine} using dielectric of "
                f"{self.molecule.dielectric}",
                "Density calculation complete",
            ],
            "charges": [
                f"Chargemol calculating charges using DDEC{self.molecule.ddec_version}",
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
            "torsion_test": [
                "Testing torsion single point energies",
                "Torsion testing complete",
            ],
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
            append_to_log(f"skipping stage: {start_key}")
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
            append_to_log("Starting parametrisation")

        # Parametrisation options:
        param_dict = {"antechamber": AnteChamber, "xml": XML, "openff": OpenFF}

        # If we are using xml we have to move it to QUBEKit working dir
        if molecule.parameter_engine == "xml":
            xml_name = f"{molecule.name}.xml"
            if xml_name not in os.listdir("."):
                try:
                    copy(
                        os.path.join(molecule.home, f"{molecule.name}.xml"),
                        f"{molecule.name}.xml",
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        """You need to supply an xml file if you wish to use xml-based parametrisation;
                                            put this file in the location you are running QUBEKit from.
                                            Alternatively, use a different parametrisation method such as:
                                            -param antechamber"""
                    )

        # Perform the parametrisation
        # If the method is none the molecule is not parameterised but the parameter holders are initiated
        if molecule.parameter_engine == "none":
            Parametrisation(molecule).gather_parameters()
        else:
            param_dict[molecule.parameter_engine](molecule)

        if verbose:
            append_to_log(
                f"Finishing parametrisation of molecule with {molecule.parameter_engine}"
            )

        return molecule

    @staticmethod
    def mm_optimise(molecule: Ligand) -> Ligand:
        """
        Use an mm force field to get the initial optimisation of a molecule

        options
        ---------
        RDKit MFF or UFF force fields can have strange effects on the geometry of molecules

        Geometric / OpenMM depends on the force field the molecule was parameterised with gaff/2, OPLS smirnoff.
        #TODO replace with a general optimiser using QCEngine.
        """

        append_to_log("Starting mm_optimisation")
        # Check which method we want then do the optimisation
        if (
            molecule.mm_opt_method == "none"
            or molecule.parameter_engine == "OpenFF_generics"
        ):
            # Skip the optimisation step
            molecule.coords["mm"] = molecule.coords["input"]

        elif molecule.mm_opt_method == "openmm":
            if molecule.parameter_engine == "none":
                raise OptimisationFailed(
                    "You cannot optimise a molecule with OpenMM and no initial parameters; "
                    "consider parametrising or using UFF/MFF in RDKit"
                )
            else:
                # Make the inputs
                molecule.write_pdb(input_type="input")
                molecule.write_parameters()
                # Run geometric
                # TODO Should this be moved to a function? Seems like a likely point of failure
                with open("log.txt", "w+") as log:
                    sp.run(
                        f"geometric-optimize --epsilon 0.0 --maxiter {molecule.iterations} --pdb "
                        f"{molecule.name}.pdb --engine openmm {molecule.name}.xml "
                        f'{molecule.constraints_file if molecule.constraints_file is not None else ""}',
                        shell=True,
                        stdout=log,
                        stderr=log,
                    )

                molecule.add_conformers(f"{molecule.name}_optim.xyz", input_type="traj")
                molecule.coords["mm"] = molecule.coords["traj"][-1]

        else:
            # TODO change to qcengine as this can already be done
            # Run an rdkit optimisation with the right FF
            rdkit_ff = {"rdkit_mff": "MFF", "rdkit_uff": "UFF"}[molecule.mm_opt_method]
            rdkit_mol = RDKit.mm_optimise(molecule.rdkit_mol, ff=rdkit_ff)
            molecule.coords["mm"] = rdkit_mol.GetConformer().GetPositions()

        append_to_log(
            f"Finishing mm_optimisation of the molecule with {molecule.mm_opt_method}"
        )

        return molecule

    def qm_optimise(self, molecule: Ligand) -> Ligand:
        """Optimise the molecule coords. Can be through PSI4 (with(out) geometric) or through Gaussian."""

        append_to_log("Starting qm_optimisation")
        MAX_RESTARTS = 3

        if molecule.geometric and (molecule.bonds_engine == "psi4"):
            qceng = QCEngine(molecule)
            result = qceng.call_qcengine(
                engine="geometric",
                driver="gradient",
                input_type=f'{"mm" if list(molecule.coords["mm"]) else "input"}',
            )

            restart_count = 0

            while (not result["success"]) and (restart_count < MAX_RESTARTS):
                append_to_log(
                    f'{molecule.bonds_engine} optimisation failed with error {result["error"]}; restarting',
                    msg_type="minor",
                )

                try:
                    molecule.coords["temp"] = np.array(
                        result["input_data"]["final_molecule"]["geometry"]
                    ).reshape((len(molecule.atoms), 3))
                    molecule.coords["temp"] *= constants.BOHR_TO_ANGS

                    result = qceng.call_qcengine(
                        engine="geometric", driver="gradient", input_type="temp"
                    )

                except (KeyError, TypeError):
                    result = qceng.call_qcengine(
                        engine="geometric",
                        driver="gradient",
                        input_type=f'{"mm" if list(molecule.coords["mm"]) else "input"}',
                    )

                restart_count += 1

            if not result["success"]:
                raise OptimisationFailed("The optimisation did not converge")

            molecule.read_geometric_traj(result["trajectory"])

            # store the final molecule as the qm optimised structure
            molecule.coords["qm"] = np.array(
                result["final_molecule"]["geometry"]
            ).reshape((len(molecule.atoms), 3))
            molecule.coords["qm"] *= constants.BOHR_TO_ANGS

            molecule.qm_energy = result["energies"][-1]

            # Write out the trajectory file
            molecule.write_xyz("traj", name=f"{molecule.name}_opt")
            molecule.write_xyz("qm", name="opt")

        # Using Gaussian or geometric off
        else:
            qm_engine = self.engine_dict[molecule.bonds_engine](molecule)
            result = qm_engine.generate_input(
                input_type=f'{"mm" if list(molecule.coords["mm"]) else "input"}',
                optimise=True,
                execute=molecule.bonds_engine,
            )

            restart_count = 0
            while (not result["success"]) and (restart_count < MAX_RESTARTS):
                append_to_log(
                    f'{molecule.bonds_engine} optimisation failed with error {result["error"]}; restarting',
                    msg_type="minor",
                )

                if result["error"] == "FileIO":
                    result = qm_engine.generate_input(
                        "mm", optimise=True, restart=True, execute=molecule.bonds_engine
                    )
                elif result["error"] == "Max iterations":
                    result = qm_engine.generate_input(
                        "input",
                        optimise=True,
                        restart=True,
                        execute=molecule.bonds_engine,
                    )

                else:
                    molecule.coords["temp"] = RDKit.generate_conformers(
                        molecule.rdkit_mol
                    )[-1]
                    result = qm_engine.generate_input(
                        "temp", optimise=True, execute=molecule.bonds_engine
                    )

                restart_count += 1

            if not result["success"]:
                raise OptimisationFailed(
                    f"{molecule.bonds_engine} optimisation did not converge after 3 restarts; "
                    f"last error {result['error']}"
                )

            molecule.coords["qm"], molecule.qm_energy = qm_engine.optimised_structure()
            molecule.write_xyz("qm", name="opt")

        append_to_log(
            f'Finishing qm_optimisation of molecule{" using geometric" if molecule.geometric else ""}'
        )

        return molecule

    @staticmethod
    def hessian(molecule: Ligand) -> Ligand:
        """Using the assigned bonds engine, calculate and extract the Hessian matrix."""

        append_to_log("Starting hessian calculation")

        if molecule.bonds_engine in ["g09", "g16"]:
            qm_engine = Gaussian(molecule)

            # Use the checkpoint file as this has higher xyz precision
            try:
                copy(
                    os.path.join(molecule.home, "03_qm_optimise", "lig.chk"), "lig.chk"
                )
                result = qm_engine.generate_input(
                    "qm", hessian=True, restart=True, execute=molecule.bonds_engine
                )
            except FileNotFoundError:
                append_to_log(
                    "qm_optimise checkpoint not found, optimising first to refine atomic coordinates",
                    msg_type="minor",
                )
                result = qm_engine.generate_input(
                    "qm", optimise=True, hessian=True, execute=molecule.bonds_engine
                )

            if not result["success"]:
                raise HessianCalculationFailed(
                    "The hessian was not calculated check the log file."
                )

            hessian = qm_engine.hessian()

        else:
            hessian = QCEngine(molecule).call_qcengine(
                engine="psi4", driver="hessian", input_type="qm"
            )
            np.savetxt("hessian.txt", hessian)

        molecule.hessian = hessian

        append_to_log(f"Finishing Hessian calculation using {molecule.bonds_engine}")

        return molecule

    @staticmethod
    def mod_sem(molecule: Ligand) -> Ligand:
        """Modified Seminario for bonds and angles."""

        append_to_log("Starting mod_Seminario method")

        mod_sem = ModSeminario(molecule)

        mod_sem.modified_seminario_method()
        if molecule.enable_symmetry:
            mod_sem.symmetrise_bonded_parameters()

        append_to_log("Finishing Mod_Seminario method")

        return molecule

    def density(self, molecule):
        """Perform density calculation with the qm engine."""

        append_to_log("Starting density calculation")

        if molecule.density_engine == "onetep":
            molecule.write_xyz(input_type="qm")
            # If using ONETEP, stop after this step
            append_to_log("Density analysis file made for ONETEP")

            # Edit the order to end here
            self.order = OrderedDict(
                [
                    ("density", self.density),
                    ("charges", self.skip),
                    ("lennard_jones", self.skip),
                    ("torsion_scan", self.skip),
                    ("pause", self.pause),
                    ("finalise", self.finalise),
                ]
            )

        else:
            qm_engine = self.engine_dict[molecule.density_engine](molecule)
            qm_engine.generate_input(
                input_type="qm" if list(molecule.coords["qm"]) else "input",
                density=True,
                execute=molecule.density_engine,
            )
            append_to_log("Finishing Density calculation")

        return molecule

    @staticmethod
    def charges(molecule):
        """Perform DDEC calculation with Chargemol."""

        append_to_log("Starting charge partitioning")
        copy(
            os.path.join(molecule.home, "06_density", f"{molecule.name}.wfx"),
            f"{molecule.name}.wfx",
        )
        c_mol = Chargemol(molecule)
        c_mol.generate_input()

        ExtractChargeData(molecule).extract_charge_data()

        append_to_log(
            f"Finishing charge partitioning with Chargemol and DDEC{molecule.ddec_version}"
        )
        append_to_log("Starting virtual sites calculation")

        if molecule.enable_virtual_sites:
            vs = VirtualSites(molecule)
            vs.calculate_virtual_sites()

            # Find extra site positions in local coords if present and tweak the charges of the parent
            if molecule.charges_engine == "onetep":
                extract_extra_sites_onetep(molecule)

        # Ensure the net charge is an integer value and adds up to molecule.charge
        fix_net_charge(molecule)

        append_to_log("Finishing virtual sites calculation")

        return molecule

    @staticmethod
    def lennard_jones(molecule):
        """Calculate Lennard-Jones parameters, and extract virtual sites."""

        append_to_log("Starting Lennard-Jones parameter calculation")

        LennardJones(molecule).calculate_non_bonded_force()

        append_to_log("Finishing Lennard-Jones parameter calculation")

        return molecule

    @staticmethod
    def torsion_scan(molecule):
        """Perform torsion scan."""

        append_to_log("Starting torsion_scans")

        tor_scan = TorsionScan(molecule)

        # Check that we have a scan order for the molecule this should of been captured from the dihedral file
        tor_scan.find_scan_order()
        tor_scan.scan()

        append_to_log("Finishing torsion_scans")

        return molecule

    @staticmethod
    def torsion_optimise(molecule):
        """Perform torsion optimisation."""

        append_to_log("Starting torsion_optimisations")

        # First we should make sure we have collected the results of the scans
        if molecule.qm_scans is None:
            os.chdir(os.path.join(molecule.home, "09_torsion_scan"))
            scan = TorsionScan(molecule)
            if molecule.scan_order is None:
                scan.find_scan_order()
            scan.collect_scan()
            os.chdir(os.path.join(molecule.home, "10_torsion_optimise"))

        TorsionOptimiser(molecule).run()

        append_to_log("Finishing torsion_optimisations")

        return molecule

    @staticmethod
    def finalise(molecule):
        """
        Make the xml and pdb file;
        print the ligand object to terminal (in abbreviated form) and to the log file (unabbreviated).
        """

        molecule.write_pdb()
        molecule.write_parameters()

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
            "density folder and use QUBEKit -restart lennard_jones."
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

    @staticmethod
    def torsion_test(molecule):
        """Take the molecule and do the torsion test method."""

        # If there is a constraints file we should move it
        if molecule.constraints_file is not None:
            copy(molecule.constraints_file, molecule.constraints_file.name)

        TorsionOptimiser(molecule).torsion_test()

        printf("Torsion testing done!")
        delattr(molecule, "torsion_test")

        return molecule


def main():
    """Putting class call inside a function squashes the __repr__ call"""

    ArgsAndConfigs()


if __name__ == "__main__":
    # For running with debugger, normal entry point is defined in setup.py
    main()
