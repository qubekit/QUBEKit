#!/usr/bin/env python3

from QUBEKit.ligand import Ligand
from QUBEKit.utils.datastructures import Atom, CustomNamespace
from QUBEKit.utils.file_handling import ExtractChargeData

import networkx as nx
import numpy as np
import pytest


class Water:
    def __init__(self):
        self.name = 'water'
        self.atoms = [
            Atom(8, 'O', 0, -0.827099, [1, 2]),
            Atom(1, 'H', 1, 0.41355, [0]),
            Atom(1, 'H', 1, 0.41355, [0])]

        self.coords = {
            'qm': np.array([[-0.00191868, 0.38989824, 0.],
                            [-0.7626204, -0.19870548, 0.],
                            [0.76453909, -0.19119276, 0.]])}

        self.topology = nx.Graph()
        self.topology.add_node(0)
        self.topology.add_node(1)
        self.topology.add_node(2)
        self.topology.add_edge(0, 1)
        self.topology.add_edge(0, 2)

        self.ddec_data = {
            0: CustomNamespace(a_i=44312.906375462444, atomic_symbol='O', b_i=47.04465571009466, charge=-0.827099,
                               r_aim=1.7571614241044191, volume=29.273005),
            1: CustomNamespace(a_i=0.0, atomic_symbol='H', b_i=0, charge=0.41355, r_aim=0.6833737065249833,
                               volume=2.425428),
            2: CustomNamespace(a_i=0.0, atomic_symbol='H', b_i=0, charge=0.413549, r_aim=0.6833737065249833,
                               volume=2.425428)}

        self.dipole_moment_data = {
            0: CustomNamespace(x_dipole=-0.000457, y_dipole=0.021382, z_dipole=0.0),
            1: CustomNamespace(x_dipole=-0.034451, y_dipole=-0.010667, z_dipole=0.0),
            2: CustomNamespace(x_dipole=0.03439, y_dipole=-0.010372, z_dipole=-0.0)}

        self.quadrupole_moment_data = {
            0: CustomNamespace(q_3z2_r2=-0.786822, q_x2_y2=0.307273, q_xy=0.001842, q_xz=-0.0, q_yz=0.0),
            1: CustomNamespace(q_3z2_r2=-0.097215, q_x2_y2=0.018178, q_xy=0.001573, q_xz=0.0, q_yz=0.0),
            2: CustomNamespace(q_3z2_r2=-0.097264, q_x2_y2=0.018224, q_xy=-0.001287, q_xz=-0.0, q_yz=-0.0)}

        self.cloud_pen_data = {
            0: CustomNamespace(a=-0.126185, atomic_symbol='O', b=2.066872),
            1: CustomNamespace(a=-2.638211, atomic_symbol='H', b=1.917509),
            2: CustomNamespace(a=-2.627835, atomic_symbol='H', b=1.920499)}

        self.enable_symmetry = True
        self.v_site_error_factor = 1.005


def load_molecule():
    molecule = Ligand('CO', 'methanol')
    ExtractChargeData(molecule).extract_charge_data()
