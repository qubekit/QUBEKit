#!/usr/bin/env python3

from QUBEKit.utils.constants import ANGS_TO_BOHR

from multiprocessing import current_process, Process, Queue
import os
import queue

from geometric.engine import OpenMM
from geometric.internal import DelocalizedInternalCoordinates
from geometric.molecule import Molecule
from geometric.optimize import Optimize, OptParams, ParseConstraints
from geometric.run_json import make_constraints_string

from torsiondrive import td_api

import numpy as np

# TODO OpenMM does not respect the number of designated threads so every task tries to run on every core


class TorsionDriveController:
    """

    """

    def __init__(self, molecule, bond, controller):

        self.molecule = molecule
        self.dihedrals = [[i for i in molecule.dihedrals[bond][0]]]
        self.grid_spacing = [molecule.increment]
        self.elements = [atom.atomic_symbol for atom in molecule.atoms]
        self.init_coords = [(molecule.coords['input'] * ANGS_TO_BOHR).ravel().tolist()]
        self.dihedral_ranges = [[molecule.dih_start, molecule.dih_end]]
        self.energy_decrease_thresh = None
        self.energy_upper_limit = None
        self.td_state = None
        self.threads = molecule.threads
        self.next_jobs = Queue()
        self.job_results = None
        self.results = None
        self.geo_mol = Molecule(f'{molecule.name}.pdb')
        self.params = None
        self.update_params()
        self.create_state()
        os.environ['OPENMM_CPU_THREADS'] = '1'

    def update_params(self):
        """Unpack all params which need to be updated."""

        params = {'qccnv': True, 'reset': True, 'enforce': 0.0, 'epsilon': 0}
        self.params = OptParams(**params)

    def create_state(self):
        """

        :return:
        """

        self.td_state = td_api.create_initial_state(
            dihedrals=self.dihedrals,
            grid_spacing=self.grid_spacing,
            init_coords=self.init_coords,
            elements=self.elements,
            dihedral_ranges=self.dihedral_ranges,
            energy_decrease_thresh=self.energy_decrease_thresh,
            energy_upper_limit=self.energy_upper_limit,
        )

    def get_next_jobs(self, next_jobs):
        """

        :param next_jobs:
        :return:
        """

        new_jobs = td_api.next_jobs_from_state(self.td_state, verbose=True)
        # Now put them into the queue
        for grid_id_str, job_geo_list in new_jobs.items():
            for job_geo in job_geo_list:
                dihedral_values = td_api.grid_id_from_string(grid_id_str)
                next_jobs.put((np.array(job_geo), dihedral_values[0]))

        return next_jobs

    def create_engine(self, engine):
        """

        :param engine:
        :return:
        """

        engine_actions = {
            'openmm': self._make_openmm_engine(),
            'psi4': self._make_psi4_engine(),
            'gaussian': self._make_gaussian_engine()
        }

        try:
            engine_actions.get(engine)
        except KeyError:
            raise NotImplementedError('Please use openmm, psi4 or gaussian.')

    def create_diheral_constraints(self, angle):
        constraints_dict = {'set': [{'type': 'dihedral', 'indices': self.dihedrals[0], 'value': angle}]}
        constraints_string = make_constraints_string(constraints_dict)

        return ParseConstraints(self.geo_mol, constraints_string)

    def _make_openmm_engine(self):
        return OpenMM(self.geo_mol, f'{self.molecule.name}.pdb', f'{self.molecule.name}.xml')

    def _make_gaussian_engine(self):
        pass

    def _make_psi4_engine(self):
        pass

    def optimise_grid_point(self, working_queue, output_queue):
        """Consume a grid point job from the queue and optimise"""

        # os.environ['OPENMM_CPU_THREADS'] = '1'
        # print(os.environ['OPENMM_CPU_THREADS'])
        while True:
            try:
                inputs = working_queue.get_nowait()

            except queue.Empty:
                # print('no job in queue')
                break
            else:
                # try to compute the optimised point
                # print(f'current optimisation point at {inputs[1]} performed by thread id {current_process().name}')
                opt = self.create_optimiser(inputs[0], inputs[1])

                # now put the results in the finished queue
                final_geo = opt.xyzs[-1]
                final_energy = opt.Data['qm_energies'][-1]
                # print(f'opt done! {current_process().name}')
                output_queue.put((inputs[1], inputs[0], final_geo, final_energy))
                # print(f'optimisation finished at {inputs[1]} by thread id {current_process().name}')

    def create_optimiser(self, coordinates, dihedral_angle):
        """Instance and set up the geometric optimiser"""

        print(f'{current_process().name} {os.environ["OPENMM_CPU_THREADS"]}')
        cons, cvals = self.create_diheral_constraints(dihedral_angle)
        engine = self._make_openmm_engine()

        # make the coord system using default values for now
        coord_class, connect, add_cart = DelocalizedInternalCoordinates, False, False

        ic = coord_class(self.geo_mol, build=True, connect=connect, addcart=add_cart, constraints=cons,
                         cvals=cvals[0] or None)

        return Optimize(coordinates, self.geo_mol, ic, engine, 'test', self.params)

    def update_state(self, output_queue):
        """Using the output queue, update the tdrive state"""

        job_results = {}
        while not output_queue.empty():
            results = output_queue.get_nowait()
            job_results.setdefault(str(results[0]), []).append(*results[1:])

        td_api.update_state(self.td_state, job_results)

    def start_server(self):

        working_queue = Queue()
        output_queue = Queue()
        # TODO Safe/guaranteed exit condition?
        while True:
            # Start by getting the next jobs
            working_queue = self.get_next_jobs(working_queue)

            if not working_queue.qsize():
                return td_api.collect_lowest_energies(self.td_state)

            # Now we know the grid point to run set up the optimiser class
            processors = []
            for i in range(4):
                p = Process(target=self.optimise_grid_point, args=(working_queue, output_queue))
                processors.append(p)
                p.start()

            for p in processors:
                p.join()

            # print('waiting for all jobs to finish')

            self.update_state(output_queue)

    def run_tdrive(self):
        """The access point to control the class will collect the optimised energies and structures
        in to the right order"""

        self.results = {}
        optimised_energies = self.start_server()
        # now sort the energies by key
        for items in sorted(optimised_energies.items()):
            self.results[items[0]] = [items[1]]

        # now we can collect the optimised geometries from the state and convert them to ang
        for key, value in self.td_state['grid_status'].items():
            grid_id = td_api.grid_id_from_string(key)
            for results in value:
                if self.results[grid_id][0] == results[-1]:
                    self.results[grid_id].append(results[1])
