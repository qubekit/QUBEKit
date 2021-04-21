#!/usr/bin/env python3

"""
Purpose of this file is to read various inputs and produce the info required for
    Ligand() or Protein()
Should be very little calculation here, simply file reading and some small validations / checks
"""
import contextlib
import os


def make_and_change_into(name: str):
    """
    - Attempt to make a directory with name <name>, don't fail if it exists.
    - Change into the directory.
    """

    try:
        os.mkdir(name)
    except FileExistsError:
        pass
    finally:
        os.chdir(name)


@contextlib.contextmanager
def folder_setup(folder_name: str):
    """
    A helper function to make a directory with name and move into it and move back to the starting location when
    finished.
    """
    cwd = os.getcwd()
    make_and_change_into(folder_name)
    yield
    os.chdir(cwd)


def get_data(relative_path: str) -> str:
    """
    Get the file path to some data in the qubekit package.
    Parameters
    ----------
    relative_path
        The relative path to the file that should be loaded from QUBEKit/data.
    Returns
    -------
    str
        The absolute path to the requested file.
    """
    from pkg_resources import resource_filename

    fn = resource_filename("qubekit", os.path.join("data", relative_path))
    if not os.path.exists(fn):
        raise ValueError(
            f"{relative_path} does not exist. If you have just added it, you'll have to re-install"
        )

    return fn
