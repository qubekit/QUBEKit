#!/usr/bin/env python3


class HessianCalculationFailed(Exception):
    """ """


class TorsionDriveFailed(Exception):
    """ """


class PickleFileNotFound(Exception):
    """
    Cannot find .QUBEKit_states.
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


class MissingParameterError(Exception):
    """
    When no parameter can be found for the given interacting atoms.
    """


class MissingReferenceData(Exception):
    """
    Some reference data needed for an operation is missing.
    """


class TorsionDriveDataError(Exception):
    """
    Raised when a torsiondrivedata object has an inconsistency.
    """


class ForceBalanceError(Exception):
    """
    This indicates there was an error while using forcebalance to optimise some parameters.
    """


class GeometryOptimisationError(Exception):
    """
    The molecule conformer could not be optimised.
    """


class WorkFlowExecutionError(Exception):
    """
    Raised whenever we hit an error executing the workflow.
    """
