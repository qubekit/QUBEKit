#!/usr/bin/env python3

from QUBEKit.ligand import Ligand
from QUBEKit.utils.file_handling import ExtractChargeData

import pytest


def load_molecule():
    molecule = Ligand('CO', 'methanol')
    ExtractChargeData(molecule).extract_charge_data()
