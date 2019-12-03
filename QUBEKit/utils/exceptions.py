#!/usr/bin/env python3


class OptimisationFailed(Exception):
    """
    Raise for seg faults from PSI4 - geomeTRIC/Torsiondrive/QCEngine interactions.
    This should mean it's more obvious to users when there's a segfault.
    """

    pass


class HessianCalculationFailed(Exception):
    """

    """

    pass


class TorsionDriveFailed(Exception):
    """

    """

    pass


class PickleFileNotFound(Exception):
    """
    Cannot find .QUBEKit_states.
    """

    pass


class QUBEKitLogFileNotFound(Exception):
    """
    Cannot find QUBEKit_log.txt. This is only raised when a recursive search fails.
    """

    pass


class FileTypeError(Exception):
    """
    Invalid file type e.g. trying to read a mol file when we only accept pdb or mol2.
    """

    pass


class TopologyMismatch(Exception):
    """
    This indicates that the topology of a file does not match the stored topology.
    """

    pass


class ChargemolError(Exception):
    """
    Chargemol did not execute properly.
    """

    pass


class PSI4Error(Exception):
    """
    PSI4 did not execute properly
    """

    pass
