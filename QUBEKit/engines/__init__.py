#!/usr/bin/env python3

# TODO Add failed to import messages
#   Maybe separate ONETEP class into a couple of helpers as the use is fairly agnostic.

from .babel import Babel
from .chargemol import Chargemol


def missing_import(name, fail_msg=''):
    """
    Generates a class which raises an import error when initialised.
    e.g. SomeClass = missing_import('SomeClass') will make SomeClass() raise ImportError
    """
    def init(self, *args, **kwargs):
        raise ImportError(
            f'The class {name} you tried to call is not importable; '
            f'this is likely due to it not doing installed. '
            f'{f"Fail Message: {fail_msg}" if fail_msg else ""}'
        )
    return type(name, (), {'__init__': init})


try:
    from .gaussian import Gaussian
except ImportError:
    Gaussian = missing_import('Gaussian')

try:
    from .onetep import ONETEP
except ImportError:
    ONETEP = missing_import('ONETEP')

try:
    from .openmm import OpenMM
except ImportError:
    OpenMM = missing_import('OpenMM')

try:
    from .psi4 import PSI4
except ImportError:
    PSI4 = missing_import('PSI4')

try:
    from .qcengine import QCEngine
except ImportError:
    QCEngine = missing_import('QCEngine')

try:
    from .rdkit import RDKit
except ImportError:
    RDKit = missing_import('RDKit')
