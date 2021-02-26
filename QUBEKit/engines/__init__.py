#!/usr/bin/env python3
"""
Ensures that errors with particular engines are only raised if said engine is being used.
No point raising "cannot execute psi4 command" if user is using g09. See try_load docstring for info.
"""
from QUBEKit.utils.helpers import try_load

from .chargemol import Chargemol
from .gaussian import Gaussian
from .geometry_optimiser import GeometryOptimiser

OpenMM = try_load("OpenMM", "QUBEKit.engines.openmm")
PSI4 = try_load("PSI4", "QUBEKit.engines.psi4")
QCEngine = try_load("QCEngine", "QUBEKit.engines.qcengine")
