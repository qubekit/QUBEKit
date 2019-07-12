#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils.decorators import for_all_methods, timer_logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull


# TODO do we need this class anymore? It only makes an xyz file; maybe move calculate_hull to helpers
@for_all_methods(timer_logger)
class ONETEP(Engines):

    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_input(self, input_type='input', density=False):
        """ONETEP takes a xyz input file."""

        if density:
            self.molecule.write_xyz(input_type=input_type)

        # should we make a onetep run file? this is quite specific?
        print('Run this file in ONETEP.')

    def calculate_hull(self):
        """
        Generate the smallest convex hull which encloses the molecule.
        Then make a 3d plot of the points and hull.
        """

        coords = np.array([atom[1:] for atom in self.molecule.coords['input']])

        hull = ConvexHull(coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        ax.plot(coords.T[0], coords.T[1], coords.T[2], 'ko')

        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(coords[simplex, 0], coords[simplex, 1], coords[simplex, 2], color='lightseagreen')

        plt.show()
