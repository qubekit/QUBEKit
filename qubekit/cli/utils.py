from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterAttribute,
    vdWHandler,
)
from openmm import unit


class QUBEKitHandler(vdWHandler):
    """A plugin handler to enable the fitting of Rfree parameters using evaluator"""

    _TAGNAME = "QUBEKitvdW"
    hfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    xfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    cfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    nfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    ofree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    clfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    sfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    ffree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    brfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    pfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    ifree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    bfree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    sifree = ParameterAttribute(0, converter=float, unit=unit.angstroms)
    alpha = ParameterAttribute(1, converter=float, unit=unit.angstroms)
    beta = ParameterAttribute(0, converter=float, unit=unit.angstroms)
