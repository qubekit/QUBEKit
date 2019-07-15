#!/usr/bin/env python3

from QUBEKit.utils import constants
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.helpers import append_to_log

import numpy as np

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

import xml.etree.ElementTree as ET

from copy import deepcopy


@for_all_methods(timer_logger)
class OpenMM:
    """This class acts as a wrapper around OpenMM so we can many basic functions using the class"""

    def __init__(self, molecule):
        self.molecule = molecule
        self.system = None
        self.simulation = None
        self.combination = None
        self.pdb = molecule.name + '.pdb'
        self.xml = molecule.name + '.xml'
        self.openmm_system()

    def openmm_system(self):
        """Initialise the OpenMM system we will use to evaluate the energies."""

        # Load the initial coords into the system and initialise
        pdb = app.PDBFile(self.pdb)
        forcefield = app.ForceField(self.xml)
        modeller = app.Modeller(pdb.topology, pdb.positions)  # set the initial positions from the pdb
        self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        # Check what combination rule we should be using from the xml
        xmlstr = open(self.xml).read()
        # check if we have opls combination rules if the xml is present
        try:
            self.combination = ET.fromstring(xmlstr).find('NonbondedForce').attrib['combination']
            append_to_log('OPLS combination rules found in xml file', msg_type='minor')
        except AttributeError:
            pass
        except KeyError:
            pass

        if self.combination == 'opls':
            self.opls_lj()

        temperature = constants.STP * unit.kelvin
        integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)

        self.simulation = app.Simulation(modeller.topology, self.system, integrator)
        self.simulation.context.setPositions(modeller.positions)

    def get_energy(self, position):
        """
        Return the MM calculated energy of the structure
        :param position: The OpenMM formatted atomic positions
        :param forces: If we should also get the forces
        :return:
        """

        # update the positions of the system
        self.simulation.context.setPositions(position)

        # Get the energy from the new state
        state = self.simulation.context.getState(getEnergy=True)

        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        return energy

    def opls_lj(self):
        """
        This function changes the standard OpenMM combination rules to use OPLS, execp and normal pairs are only
        required if their are virtual sites in the molecule.
        """

        # Get the system information from the openmm system
        forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in
                  range(self.system.getNumForces())}
        # Use the nondonded_force to get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = mm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
        lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        self.system.addForce(lorentz)

        l_j_set = {}
        # For each particle, calculate the combination list again
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            l_j_set[index] = (sigma, epsilon, charge)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(index, charge, 0, 0)

        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                charge = 0.5 * (l_j_set[p1][2] * l_j_set[p2][2])
                sig14 = np.sqrt(l_j_set[p1][0] * l_j_set[p2][0])
                nonbonded_force.setExceptionParameters(i, p1, p2, charge, sig14, eps)
            # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
            # Old method which needs updating
            # if excep_pairs:
            #     for x in range(len(excep_pairs)):  # scale 14 interactions
            #         if p1 == excep_pairs[x, 0] and p2 == excep_pairs[x, 1] or p2 == excep_pairs[x, 0] and p1 == \
            #                 excep_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2 * 0.5
            #             sig14 = sqrt(sigma1 * sigma2) * 0.5
            #             eps = sqrt(epsilon1 * epsilon2) * 0.5
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            #
            # if normal_pairs:
            #     for x in range(len(normal_pairs)):
            #         if p1 == normal_pairs[x, 0] and p2 == normal_pairs[x, 1] or p2 == normal_pairs[x, 0] and p1 == \
            #                 normal_pairs[x, 1]:
            #             charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
            #             charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
            #             q = charge1 * charge2
            #             sig14 = sqrt(sigma1 * sigma2)
            #             eps = sqrt(epsilon1 * epsilon2)
            #             nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

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

    def calculate_hessian(self, finite_step):
        """
        Using finite displacement calculate the hessian matrix of the molecule using symmetric difference quotient (SQD) rule.
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
                # Do less energy evaluations on the diagonal of the matrix
                if i == j:
                    coords = deepcopy(input_coords)
                    coords[i] += 2 * finite_step
                    e1 = self.get_energy(self.format_coords(coords))
                    coords = deepcopy(input_coords)
                    coords[i] -= 2 * finite_step
                    e2 = self.get_energy(self.format_coords(coords))
                    hessian[i, j] = (e1 + e2) / (4 * finite_step**2 * self.molecule.atoms[i // 3].mass)
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
                    hessian[i, j] = (e1 + e2 - e3 - e4) / (4 * finite_step ** 2 * self.molecule.atoms[i // 3].mass)

        # Now make the matrix symmetric
        sym_hessian = hessian + hessian.T - np.diag(hessian.diagonal())
        return sym_hessian

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
