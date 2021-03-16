#!/usr/bin/env python3


class OptimisationFailed(Exception):
    """
    Raise for seg faults from PSI4 - geomeTRIC/Torsiondrive/QCEngine interactions.
    This should mean it's more obvious to users when there's a segfault.
    """


class HessianCalculationFailed(Exception):
    """"""


class TorsionDriveFailed(Exception):
    """"""


class PickleFileNotFound(Exception):
    """
    Cannot find .QUBEKit_states.
    """


class QUBEKitLogFileNotFound(Exception):
    """
    Cannot find QUBEKit_log.txt. This is only raised when a recursive search fails.
    """


class FileTypeError(Exception):
    """
    Invalid file type e.g. trying to read a mol file when we only accept pdb or mol2.
    """


class TopologyMismatch(Exception):
    """
    This indicates that the topology of a file does not match the stored topology.
    """


class ChargemolError(Exception):
    """
    Chargemol did not execute properly.
    """


class PSI4Error(Exception):
    """
    PSI4 did not execute properly
    """


class SmartsError(Exception):
    """
    The requested smarts could not be interpreted by rdkit
    """


class SpecificationError(Exception):
    """
    When a requested specification is not possible.
    """


class ConformerError(Exception):
    """
    When a conformation is incorrect or missing.
    """


class StereoChemistryError(Exception):
    """
    When the stereochemistry is incorrect in an rdkit instance created from a qubekit molecule.
    """
