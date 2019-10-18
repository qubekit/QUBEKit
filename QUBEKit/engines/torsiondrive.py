#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils.decorators import for_all_methods, timer_logger

import json


@for_all_methods(timer_logger)
class TorsionDrive(Engines):

    def __init__(self, molecule, dihedral, json_filename):

        super().__init__(molecule)

        self.dihedral = dihedral
        self.json_filename = json_filename

    def write_json(self):
        """
        Take all necessary info from the molecule object and format it to a json.
        There will be one json per job
        :return: json object
        """

        dihedrals = self.dihedral
        grid_spacing = []
        elements = []
        init_coords = self.molecule.coords
        grid_status = dict()

        scan_info = [dihedrals, grid_spacing, elements, init_coords, grid_status]

        with open(self.json_filename, 'w') as json_file:
            json.dump(scan_info, json_file)

    def execute(self):

        with open(self.json_filename, 'r') as json_file:
            # execute
            pass
