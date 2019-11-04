#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils import constants
from QUBEKit.utils.decorators import timer_logger

from copy import deepcopy

import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET

from simtk import openmm, unit  # Ignore unit import warnings, blame rampant misuse of import * in OpenMM
from simtk.openmm import app


class OpenMM(Engines):
    """This class acts as a wrapper around OpenMM so we can handle many basic functions using the class"""

    def __init__(self, molecule, filename=None, forcefield=None):

        super().__init__(molecule)

        self.filename = filename or f'{molecule.name}.pdb'
        self.forcefield = forcefield or f'{molecule.name}.xml'

        self.system = None
        self.simulation = None
        self.combination = None

        self.normal_pairs = None
        self.excep_pairs = None
        self.has_vsites = bool(molecule.extra_sites)

        self.create_system()

    @timer_logger
    def create_system(self):
        # set up the system using opls combo rules
        # Load the initial coords into the system and initialise
        pdb = app.PDBFile(self.filename)
        forcefield = app.ForceField(self.forcefield)
        # set the initial positions from the pdb
        modeller = app.Modeller(pdb.topology, pdb.positions)

        # if there are virtual sites we need to add them here
        try:
            self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        except ValueError:
            print('Virtual sites were found in the xml file')
            modeller.addExtraParticles(forcefield)
            self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        # Check what combination rule we should be using from the xml
        xmlstr = open(self.forcefield).read()
        # check if we have opls combination rules if the xml is present
        try:
            self.combination = ET.fromstring(xmlstr).find('NonbondedForce').attrib['combination']
        except (AttributeError, KeyError):
            pass

        # use the opls combination rules
        if self.combination == 'opls':
            print('OPLS combination rules found in XML file')
            self.opls_lj()

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        self.simulation = app.Simulation(modeller.topology, self.system, integrator, platform)
        self.simulation.context.setPositions(modeller.positions)

    # get_energy is called too many times so timer_logger decorator should not be applied.
    def get_energy(self, position):
        """
        Return the MM calculated energy of the structure
        :param position: The OpenMM formatted atomic positions
        :return: energy: vacuum energy state of the system
        """

        # update the positions of the system
        self.simulation.context.setPositions(position)

        # Get the energy from the new state
        state = self.simulation.context.getState(getEnergy=True)

        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        return energy

    @timer_logger
    def opls_lj(self):
        # Get the system information from the OpenMM system
        forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in
                  range(self.system.getNumForces())}

        # Use the nonbonded_force to get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = openmm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')

        lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        self.system.addForce(lorentz)
        l_j_set = {}
        # For each particle, calculate the combination list again
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            l_j_set[index] = (charge, sigma, epsilon)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(index, charge, 0, 0)

        exclusions = {}
        for i in range(nonbonded_force.getNumExceptions()):
            p1, p2, q, _, eps = nonbonded_force.getExceptionParameters(i)
            # store the index of the exception by the sorted atom keys
            # ALL THE 12, 13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            lorentz.addExclusion(p1, p2)
            exclusions[tuple(sorted(p1, p2))] = i
            if eps._value != 0.0:
                sig14 = np.sqrt(l_j_set[p1][0] * l_j_set[p2][0])
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

        # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
        if self.has_vsites:
            # get the interaction lists
            self.excep_pairs, self.normal_pairs = self.get_vsite_interactions()

            for pair in self.excep_pairs:  # scale 14 interactions
                charge1, _, _ = l_j_set[pair[0]]
                charge2, _, _ = l_j_set[pair[1]]
                q = charge1 * charge2 * 0.5
                if pair not in exclusions:
                    lorentz.addExclusion(*pair)
                nonbonded_force.addException(*pair, q, 0, 0, True)

            for pair in self.normal_pairs:  # add the normal pairs here
                charge1, _, _ = l_j_set[pair[0]]
                charge2, _, _ = l_j_set[pair[1]]
                q = charge1 * charge2
                if pair not in exclusions:
                    lorentz.addExclusion(*pair)
                nonbonded_force.addException(*pair, q, 0, 0, True)

    @timer_logger
    def format_coords(self, coordinates):
        """
        Take the coordinates as a list and format to the OpenMM style of a list of tuples.
        :param coordinates: The flattened list of coordinates.
        :return: The OpenMM list of tuples.
        """

        coords = []
        for i in range(0, len(coordinates), 3):
            coords.append(tuple(coordinates[i:i+3]))

        return coords

    @timer_logger
    def calculate_hessian(self, finite_step):
        """
        Using finite displacement calculate the hessian matrix of the molecule
        using symmetric difference quotient (SQD) rule.
        :param finite_step: The finite step size used in the calculation in nm
        :return: A numpy array of the mass weighted hessian of size 3N*3N
        """

        # Create the OpenMM coords list from the qm coordinates and convert to nm
        input_coords = self.molecule.coords['qm'].flatten() * constants.ANGS_TO_NM

        # We get each hessian element from = [E(dx + dy) + E(-dx - dy) - E(dx - dy) - E(-dx + dy)] / 4 dx dy
        hessian = np.zeros((3 * len(self.molecule.atoms), 3 * len(self.molecule.atoms)))

        for i in range(3 * len(self.molecule.atoms)):
            for j in range(i, 3 * len(self.molecule.atoms)):
                # Mutate the atomic coords
                # Do fewer energy evaluations on the diagonal of the matrix
                if i == j:
                    coords = deepcopy(input_coords)
                    coords[i] += 2 * finite_step
                    e1 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= 2 * finite_step
                    e2 = self.get_energy(self.format_coords(coords))
                    hessian[i, j] = (e1 + e2) / (4 * finite_step ** 2 * self.molecule.atoms[i // 3].atomic_mass)
                else:
                    coords = deepcopy(input_coords)
                    coords[i] += finite_step
                    coords[j] += finite_step
                    e1 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= finite_step
                    coords[j] -= finite_step
                    e2 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] += finite_step
                    coords[j] -= finite_step
                    e3 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= finite_step
                    coords[j] += finite_step
                    e4 = self.get_energy(self.format_coords(coords))
                    hessian[i, j] = (e1 + e2 - e3 - e4) / (4 * finite_step ** 2 * self.molecule.atoms[i // 3].atomic_mass)

        # hessian is currently just the upper right part of the matrix
        # transpose to get the lower left, then remove the extra diagonal terms
        sym_hessian = hessian + hessian.T - np.diag(hessian.diagonal())
        return sym_hessian

    @timer_logger
    def normal_modes(self, finite_step):
        """
        Calculate the normal modes of the molecule from the hessian matrix
        :param finite_step: The finite step size used in the calculation of the matrix
        :return: A numpy array of the normal modes of the molecule
        """

        # Get the mass weighted hessian matrix in amu
        hessian = self.calculate_hessian(finite_step)

        # Now get the eigenvalues and vectors
        e_vals, e_vectors = np.linalg.eig(hessian)
        print(e_vals)
        print(e_vectors)

    @timer_logger
    def get_vsite_interactions(self):
        # use the topology map to get the vsite interaction lists
        # add a connection to the parent then generate the 1-4 list and everything higher list
        exception_pairs, normal_pairs = [], []

        for site_key, site in self.molecule.extra_sites.items():
            site_no = site_key + len(self.molecule.atoms)
            self.molecule.topology.add_node(site_no)
            self.molecule.topology.add_edge(site_no, site[0][0])  # parent at [0][0]

        # now that all sites are in the topology we need to work out all pairs
        for site_key, site in self.molecule.extra_sites.items():
            site_no = site_key + len(self.molecule.atoms)
            path_lengths = nx.single_source_shortest_path_length(self.molecule.topology, site_no)

            for atom, length in path_lengths.items():
                if length == 3:
                    exception_pairs.append(tuple(sorted([site_no, atom])))
                elif length > 3:
                    normal_pairs.append(tuple(sorted([site_no, atom])))

        # return sets to remove the copies
        return set(exception_pairs), set(normal_pairs)
