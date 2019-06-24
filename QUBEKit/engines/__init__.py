#!/usr/bin/env python3

# TODO Add failed to import messages
#   Maybe separate ONETEP class into a couple of helpers as the use is fairly agnostic.

from .babel import Babel
from .chargemol import Chargemol


# TODO Put repeated code into a function? May be overkill
try:
    from .gaussian import Gaussian
except ImportError:
    from .base_engine import Default as Gaussian
    setattr(Gaussian, 'name', 'Gaussian')

try:
    from .onetep import ONETEP
except ImportError:
    from .base_engine import Default as ONETEP
    setattr(ONETEP, 'name', 'ONETEP')

try:
    from .openmm import OpenMM
except ImportError:
    from .base_engine import Default as OpenMM
    setattr(OpenMM, 'name', 'OpenMM')

try:
    from .psi4 import PSI4
except ImportError:
    from .base_engine import Default as PSI4
    setattr(PSI4, 'name', 'PSI4')

try:
    from .qcengine import QCEngine
except ImportError:
    from .base_engine import Default as QCEngine
    setattr(QCEngine, 'name', 'QCEngine')

try:
    from .rdkit import RDKit
except ImportError:
    from .base_engine import Default as RDKit
    setattr(RDKit, 'name', 'RDKit')
