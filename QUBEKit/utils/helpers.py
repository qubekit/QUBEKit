#!/usr/bin/env python3

import csv
import decimal
import logging
import math
import operator
import os
import pickle
from collections import OrderedDict
from contextlib import contextmanager
from importlib import import_module
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from QUBEKit.utils import constants
from QUBEKit.utils.constants import COLOURS
from QUBEKit.utils.exceptions import PickleFileNotFound, TopologyMismatch

if TYPE_CHECKING:
    from QUBEKit.molecules import Ligand

# TODO Move csv stuff for bulk runs to file_handling.py


def mol_data_from_csv(csv_name: str):
    """
    Scan the csv file to find the row with the desired molecule data.
    Returns a dictionary of dictionaries in the form:
    {'methane': {'smiles': 'C', 'multiplicity': 1, ...}, 'ethane': {'smiles': 'C', ...}, ...}
    """

    with open(csv_name, "r") as csv_file:

        mol_confs = csv.DictReader(csv_file)

        rows = []
        for row in mol_confs:

            # Converts to ordinary dict rather than ordered.
            row = dict(row)
            # If there is no config given assume its the default
            row["smiles"] = row["smiles"] if row["smiles"] else None
            row["multiplicity"] = (
                int(float(row["multiplicity"])) if row["multiplicity"] else 1
            )
            row["config_file"] = (
                row["config_file"] if row["config_file"] else "default_config"
            )
            row["restart"] = row["restart"] if row["restart"] else None
            row["end"] = row["end"] if row["end"] else "finalise"
            rows.append(row)

    # Creates the nested dictionaries with the names as the keys
    final = {row["name"]: row for row in rows}

    # Removes the names from the sub-dictionaries:
    # e.g. {'methane': {'name': 'methane', 'smiles': 'C', ...}, ...}
    # ---> {'methane': {'smiles': 'C', ...}, ...}
    for val in final.values():
        del val["name"]

    return final


def generate_bulk_csv(csv_name, max_execs=None):
    """
    Generates a csv with minimal information inside.
    Contains only headers and a row of defaults and populates all of the named files where available.
    For example, 10 pdb files with a value of max_execs=6 will generate two csv files,
    one containing 6 of those files, the other with the remaining 4.

    :param csv_name: (str) name of the csv file being generated
    :param max_execs: (int or None) determines the max number of molecules--and therefore executions--per csv file.
    """

    if csv_name[-4:] != ".csv":
        raise TypeError("Invalid or unspecified file type. File must be .csv")

    # Find any local pdb files to write sample configs
    files = []
    for file in os.listdir("."):
        if file.endswith(".pdb"):
            files.append(file[:-4])

    # If max number of pdbs per file is unspecified, just put them all in one file.
    if max_execs is None:
        with open(csv_name, "w") as csv_file:

            file_writer = csv.writer(
                csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            file_writer.writerow(
                [
                    "name",
                    "smiles",
                    "multiplicity",
                    "config_file",
                    "restart",
                    "end",
                ]
            )
            for file in files:
                file_writer.writerow([file, "", 1, "", "", ""])
        print(f"{csv_name} generated.", flush=True)
        return

    try:
        max_execs = int(max_execs)
    except TypeError:
        raise TypeError(
            "Number of executions must be provided as an int greater than 1."
        )
    if max_execs > len(files):
        raise ValueError(
            "Number of executions cannot exceed the number of files provided."
        )

    # If max number of pdbs per file is specified, spread them across several csv files.
    num_csvs = math.ceil(len(files) / max_execs)

    for csv_count in range(num_csvs):
        with open(f"{csv_name[:-4]}_{str(csv_count).zfill(2)}.csv", "w") as csv_file:
            file_writer = csv.writer(
                csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            file_writer.writerow(
                [
                    "name",
                    "smiles",
                    "multiplicity",
                    "config_file",
                    "restart",
                    "end",
                ]
            )

            for file in files[csv_count * max_execs : (csv_count + 1) * max_execs]:
                file_writer.writerow([file, "", 1, "", "", ""])

        print(f"{csv_name[:-4]}_{str(csv_count).zfill(2)}.csv generated.", flush=True)


def append_to_log(
    log_file_path: Optional[str],
    message: str,
    major: bool = False,
    and_print: bool = False,
):
    """
    Appends a message to the log file in a specific format.
    Used for significant stages in the program such as when a stage has finished.
    Args:
        log_file_path:
            The log file path usually stored in Ligand as home. e.g. self.molecule.home
        message:
            Whatever text should be written to the log file (and printed).
        major:
            Is this a major message? Major messages are used to indicate progress of stages, and warnings.
        and_print:
            Should this be printed to terminal as well as to file?
    Returns:
        Only returns when no log file can be found.
    """
    # Check if the message is an empty string to avoid adding blank lines and extra separators
    if message:
        if and_print:
            print(message)
        if log_file_path is None:
            return
        log_file = os.path.join(log_file_path, "QUBEKit_log.txt")
        if major:
            message = message.upper()
        try:
            with open(log_file, "a") as file:
                file.write(f"\n{message}\n")
        except FileNotFoundError:
            return


def unpickle(location=None):
    """
    Unpickle the pickle file, and return an ordered dictionary of
    ligand instances--indexed by their progress.
    :param location: optional location of the pickle file .QUBEKit_states
    :return mol_states: An ordered dictionary of the states of the molecule at various stages
    (parametrise, mm_optimise, etc).
    """

    mol_states = OrderedDict()

    # unpickle the pickle jar
    # try to load a pickle file make sure to get all objects
    pickle_file = ".QUBEKit_states"

    if location is None:
        location = os.getcwd()
        while pickle_file not in os.listdir(location):
            location = os.path.split(location)[0]
            if not location:
                raise PickleFileNotFound()

    # Either location is provided or pickle file has been found in loop.
    pickle_path = os.path.join(location, pickle_file)

    with open(pickle_path, "rb") as jar:
        while True:
            try:
                mol = pickle.load(jar)
                mol_states[mol.state] = mol
            except EOFError:
                break

    return mol_states


@contextmanager
def _assert_wrapper(exception_type):
    """
    Makes assertions more informative when an Exception is thrown.
    Rather than just getting 'AssertionError' all the time, an actual named exception can be passed.
    Can be called multiple times in the same 'with' statement for the same exception type but different exceptions.

    Simple example use cases:

        with assert_wrapper(ValueError):
            assert (arg1 > 0), 'arg1 cannot be non-positive.'
            assert (arg2 != 12), 'arg2 cannot be 12.'
        with assert_wrapper(TypeError):
            assert (type(arg1) is not float), 'arg1 must not be a float.'
    """

    try:
        yield
    except AssertionError as exc:
        raise exception_type(*exc.args) from None


def check_symmetry(matrix, error=1e-5):
    """
    Check matrix is symmetric to within some error.
    :param matrix: numpy array of Hessian matrix
    :param error: allowed error between matrix elements which should be symmetric (i,j), (j,i)
    Error is raised if matrix is not symmetric within error
    """

    # Check the matrix transpose is equal to the matrix within error.
    with _assert_wrapper(ValueError):
        assert np.allclose(matrix, matrix.T, atol=error), "Matrix is not symmetric."

    print(
        f"{COLOURS.purple}Symmetry check successful. "
        f"The matrix is symmetric within an error of {error}.{COLOURS.end}"
    )


def fix_net_charge(molecule):
    """
    Ensure the total is exactly equal to the ideal net charge of the molecule.
    If net charge is not an integer value, MM simulations can (ex/im)plode.
    """

    decimal.setcontext(decimal.Context(prec=7))
    round_to = decimal.Decimal(10) ** -6

    for param in molecule.NonbondedForce:
        param.charge = decimal.Decimal(param.charge).quantize(round_to)

    atom_charges = sum(param.charge for param in molecule.NonbondedForce)
    extra = molecule.charge - atom_charges

    if molecule.extra_sites is not None:
        for site in molecule.extra_sites:
            site.charge = decimal.Decimal(site.charge).quantize(round_to)
            extra -= site.charge

    if extra:
        last_atom_index = molecule.n_atoms - 1
        molecule.NonbondedForce[(last_atom_index,)].charge += extra

    for param in molecule.NonbondedForce:
        param.charge = float(param.charge)

    if molecule.extra_sites is not None:
        for site in molecule.extra_sites:
            site.charge = float(site.charge)


def collect_archive_tdrive(tdrive_record, client):
    """
    This function takes in a QCArchive tdrive record and collects all of the final geometries and energies to be used in
    torsion fitting.
    :param tdrive_record: A QCArchive data object containing an optimisation and energy dictionary
    :param client:  A QCPortal client instance.
    :return: QUBEKit qm_scans data: list of energies and geometries [np.array(energies), [np.array(geometry)]]
    """

    # Sort the dictionary by ascending keys
    energy_dict = {
        int(key.strip("][")): value
        for key, value in tdrive_record.final_energy_dict.items()
    }
    sorted_energies = sorted(energy_dict.items(), key=operator.itemgetter(0))

    energies = np.array([x[1] for x in sorted_energies])

    geometry = []
    # Now make the optimization dict and store an array of the final geometry
    for pair in sorted_energies:
        min_energy_id = tdrive_record.minimum_positions[f"[{pair[0]}]"]
        opt_history = int(
            tdrive_record.optimization_history[f"[{pair[0]}]"][min_energy_id]
        )
        opt_struct = client.query_procedures(id=opt_history)[0]
        geometry.append(
            opt_struct.get_final_molecule().geometry * constants.BOHR_TO_ANGS
        )
        assert (
            opt_struct.get_final_energy() == pair[1]
        ), "The energies collected do not match the QCArchive minima."

    return energies, geometry


def missing_import(name, fail_msg=""):
    """
    Generates a class which raises an import error when initialised.
    e.g. SomeClass = missing_import('SomeClass') will make SomeClass() raise ImportError
    """

    def init(self, *args, **kwargs):
        raise ImportError(
            f"The class {name} you tried to call is not importable; "
            f"this is likely due to it not doing installed.\n\n"
            f'{f"Fail Message: {fail_msg}" if fail_msg else ""}'
        )

    return type(name, (), {"__init__": init})


def try_load(engine, module):
    """
    Try to load a particular engine from a module.
    If this fails, a dummy class is imported in its place with an import error raised on initialisation.

    :param engine: Name of the engine (PSI4, OpenFF, ONETEP, etc).
    :param module: Name of the QUBEKit module (.psi4, .openff, .onetep, etc).
    :return: Either the engine is imported as normal, or it is replaced with dummy class which
    just raises an import error with a message.
    """

    try:
        module = import_module(module, __name__)
        return getattr(module, engine)

    except (ModuleNotFoundError, AttributeError) as exc:
        print(
            f"{COLOURS.orange}Warning, failed to load: {engine}; continuing for now.\nReason: {exc}{COLOURS.end}\n"
        )
        return missing_import(engine, fail_msg=str(exc))


def update_ligand(restart_key, cls):
    """
    1. Create `old_mol` object by unpickling
    2. Initialise a `new_mol` with the input from `old_mol`
    3. Copy any objects in `old_mol` across to `new_mol`
            - check for attributes which have had names changed
    4. return `new_mol` as the fixed ligand object to be used from now onwards
    """

    # Get the old mol from the pickle file (probably in ../)
    old_mol = unpickle()[restart_key]
    # Initialise a new molecule based on the same mol_input that was used from before (cls = Ligand)
    try:
        new_mol = cls.from_rdkit(
            rdkit_mol=old_mol.rdkit_mol,
            name=old_mol.name,
            multiplicity=old_mol.multiplicity,
        )
        # make sure we set the qm geometry
        new_mol.coordinates = old_mol.coords["qm"]
    except AttributeError:
        new_mol = cls.from_rdkit(
            rdkit_mol=old_mol.to_rdkit(),
            name=old_mol.name,
            multiplicity=old_mol.multiplicity,
        )

    for attr, val in new_mol.__dict__.items():
        try:
            setattr(new_mol, attr, getattr(old_mol, attr))
        except AttributeError:
            setattr(new_mol, attr, val)

    return new_mol


@contextmanager
def hide_warnings():
    """
    Temporarily hide any warnings for function wrapped with this context manager.
    """
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def string_to_bool(string):
    """Convert a string to a bool for argparse use when casting to bool"""
    return string.casefold() in ["true", "t", "yes", "y"]


def check_proper_torsion(
    torsion: Tuple[int, int, int, int], molecule: "Ligand"
) -> bool:
    """
    Check that the given torsion is valid for the molecule graph.
    """
    for i in range(3):
        try:
            _ = molecule.get_bond_between(
                atom1_index=torsion[i], atom2_index=torsion[i + 1]
            )
        except TopologyMismatch:
            return False

    return True


def check_improper_torsion(
    improper: Tuple[int, int, int, int], molecule: "Ligand"
) -> Tuple[int, int, int, int]:
    """
    Check that the given improper is valid for the molecule graph.
    and always return the central bond as the first atom.
    """
    for atom_index in improper:
        try:
            atom = molecule.atoms[atom_index]
            bonded_atoms = set()
            for bonded in atom.bonds:
                bonded_atoms.add(bonded)
            if len(bonded_atoms.intersection(set(improper))) == 3:
                if improper[0] == atom_index:
                    return improper
                else:
                    return (atom_index, *atom.bonds)
        except IndexError:
            continue
    raise TopologyMismatch(f"The improper {improper} is not a valid for this molecule.")
