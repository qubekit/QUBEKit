#!/usr/bin/env python

# TODO use proper terminal printing from helpers.
# TODO Force balance testing

from QUBEKit.decorators import timer_logger, for_all_methods

from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from numpy import array, zeros, sqrt, sum, exp, round, append
from scipy.optimize import minimize

from subprocess import call as sub_call
from math import pi
from collections import OrderedDict
from copy import deepcopy
from os import chdir, mkdir

import matplotlib.pyplot as plt
import seaborn as sns


@for_all_methods(timer_logger)
class TorsionScan:
    """
    This class will take a QUBEKit molecule object and perform a torsiondrive QM (and MM if True) energy scan
    for each selected dihedral.
    """

    def __init__(self, molecule, qm_engine, config_dict, mm_engine='openmm', native_opt=False, verbose=False):

        # TODO Keep track of log file path in __init__ then change it when moving through scan folders.

        self.qm_engine = qm_engine
        self.defaults_dict, self.qm, self.fitting, self.descriptions = config_dict
        self.mm_engine = mm_engine
        self.constraints = None
        self.grid_space = self.fitting['increment']
        self.native_opt = native_opt
        self.verbose = verbose
        self.scan_mol = molecule
        self.cmd = {}
        self.find_scan_order()
        self.torsion_cmd()

    def find_scan_order(self):
        """
        Function takes the molecule and displays the rotatable central bonds,
        the user then enters the number of the torsions to be scanned in the order to be scanned.
        The molecule can also be supplied with a scan order already.
        """

        if self.scan_mol.scan_order:
            return self.scan_mol

        elif len(self.scan_mol.rotatable) == 1:
            print('One rotatable torsion found')
            self.scan_mol.scan_order = self.scan_mol.rotatable
            return self.scan_mol

        elif len(self.scan_mol.rotatable) == 0:
            print('No rotatable torsions found in the molecule')
            self.scan_mol.scan_order = []
            return self.scan_mol

        else:
            # Get the rotatable dihedrals from the molecule
            rotatable = list(self.scan_mol.rotatable)
            print('Please select the central bonds round which you wish to scan in the order to be scanned')
            print('Torsion number   Central-Bond   Representative Dihedral')
            # TODO Padding
            for i, bond in enumerate(rotatable):
                print(f'  {i + 1}                    {bond[0]}-{bond[1]}             '
                      f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][0] - 1]}-'
                      f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][1] - 1]}-'
                      f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][2] - 1]}-'
                      f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][3] - 1]}')

            scans = list(input('>'))  # Enter as a space separated list
            scans[:] = [scan for scan in scans if scan != ' ']  # remove all spaces from the scan list
            print(scans)

            scan_order = []
            # Add the rotatable dihedral keys to an array
            for scan in scans:
                scan_order.append(rotatable[int(scan) - 1])
            self.scan_mol.scan_order = scan_order

            return self.scan_mol

    def qm_scan_input(self, scan):
        """Function takes the rotatable dihedrals requested and writes a scan input file for torsiondrive."""

        with open('dihedrals.txt', 'w+') as out:

            out.write('# dihedral definition by atom indices starting from 0\n# i     j     k     l\n')
            scan_di = self.scan_mol.dihedrals[scan][0]
            out.write(f'  {scan_di[0]}     {scan_di[1]}     {scan_di[2]}     {scan_di[3]}\n')

        # TODO need to add PSI4 redundant mode selector

        if self.native_opt:
            self.qm_engine.generate_input(optimise=True, threads=True)

        else:
            self.qm_engine.geo_gradient(run=False, threads=True)

    def torsion_cmd(self):
        """Generates a command string to run torsiondrive based on the input commands for QM and MM."""

        # add the first basic command elements for QM
        cmd_qm = f'torsiondrive-launch {self.scan_mol.name}.psi4in dihedrals.txt '

        if self.grid_space:
            cmd_qm += f'-g {self.grid_space} '

        if self.qm_engine:
            cmd_qm += '-e psi4 '

        if self.native_opt:
            cmd_qm += '--native_opt '

        if self.verbose:
            cmd_qm += '-v '

        self.cmd = cmd_qm
        return self.cmd

    def get_energy(self, scan):
        """
        Extracts an array of energies from the scan results then stores it back
        into the molecule (in a dictionary) using the scan orders as the keys.
        """

        with open('scan.xyz', 'r') as scan_file:
            scan_energy = []
            for line in scan_file:
                if 'Energy ' in line:
                    scan_energy.append(float(line.split()[3]))

            self.scan_mol.QM_scan_energy[scan] = array(scan_energy)

            return self.scan_mol

    def start_scan(self):
        """Makes a folder and writes a new a dihedral input file for each scan."""
        # TODO put all the scans in a work queue so they can be performed in parallel

        for scan in self.scan_mol.scan_order:

            mkdir(f'SCAN_{scan}')
            chdir(f'SCAN_{scan}')
            mkdir('QM')
            chdir('QM')

            # now make the scan input files
            self.qm_scan_input(scan)
            sub_call(self.cmd, shell=True)
            self.get_energy(scan)
            chdir('../')


@for_all_methods(timer_logger)
class TorsionOptimiser:
    """Torsion optimiser class used to optimise dihedral parameters with a range of methods"""

    def __init__(self, molecule, qm_engine, config_dict, weight_mm=True, opls=True, use_force=False, step_size=0.002,
                 error_tol=1e-5, x_tol=1e-4, method='BFGS'):
        self.qm, self.fitting, self.descriptions = config_dict[1:]
        self.l_pen = self.fitting['l_pen']
        self.t_weight = self.fitting['t_weight']
        self.molecule = molecule
        self.qm_engine = qm_engine
        self.opls = opls
        self.weight_mm = weight_mm
        self.step_size = step_size
        self.methods = {'NM': 'Nelder-Mead', 'BFGS': 'BFGS'}    # Scipy minimisation method; BFGS with custom step size
        self.method = self.methods[method]
        self.error_tol = error_tol
        self.x_tol = x_tol
        self.energy_dict = molecule.QM_scan_energy
        self.use_Force = use_force
        self.mm_energy = []
        self.initial_energy = []
        self.scan_order = molecule.scan_order
        self.scan_coords = []
        self.atm_no = len(molecule.atom_names)
        self.system = None
        self.simulation = None
        self.target_energy = None
        self.qm_energy = None
        self.scan = None
        self.param_vector = None
        self.torsion_store = None
        self.k_b = 0.001987
        self.tor_types = OrderedDict()
        self.phases = [0, pi, 0, pi]
        self.rest_torsions()
        self.openmm_system()

    def mm_energies(self):
        """Evaluate the MM energies of the QM structures."""

        self.mm_energy = []
        for position in self.scan_coords:
            # update the positions of the system
            self.simulation.context.setPositions(position)

            # Then get the energy from the new state
            state = self.simulation.context.getState(getEnergy=True, getForces=self.use_Force)
            # print(f'{float(str(state.getPotentialEnergy())[:-6])/4.184} kcal/mol')
            self.mm_energy.append(float(str(state.getPotentialEnergy())[:-6]) / 4.184)  # convert from kJ to kcal

        return array(self.mm_energy)
        # get forces from the system
        # open_grad = state.getForces()

    @staticmethod
    def get_coords():
        """
        Read the torsion drive output file to get all of the coords in a format that can be passed to openmm
        so we can update positions in context without reloading the molecule.
        """

        scan_coords = []
        # open the torsion drive data file read all the scan coordinates
        with open('qdata.txt', 'r') as data:
            for line in data.readlines():
                if 'COORDS' in line:
                    # get the coords into a single array
                    coords = [float(x) / 10 for x in line.split()[1:]]
                    # convert to a list of tuples
                    tups = []
                    for i in range(0, len(coords), 3):
                        tups.append((coords[i], coords[i + 1], coords[i + 2]))
                    scan_coords.append(tups)

        return scan_coords

    def openmm_system(self):
        """Initialise the OpenMM system we will use to evaluate the energies."""

        # Load the initial coords into the system and initialise
        pdb = app.PDBFile(self.molecule.filename)
        forcefield = app.ForceField(f'{self.molecule.name}.xml')
        modeller = app.Modeller(pdb.topology, pdb.positions)  # set the initial positions from the pdb
        self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        if self.opls:
            self.opls_lj()

        temperature = 298.15 * unit.kelvin
        integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)

        self.simulation = app.Simulation(modeller.topology, self.system, integrator)
        self.simulation.context.setPositions(modeller.positions)

    def initial_energies(self):
        """Calculate the initial energies using the input xml."""

        # First, reset all periodic torsion terms back to their initial values
        for pos, key in enumerate(self.torsion_store):
            self.tor_types[pos] = [[key], [float(self.torsion_store[key][0][1]), float(self.torsion_store[key][1][1]),
                                           float(self.torsion_store[key][2][1]), float(self.torsion_store[key][3][1])],
                                   [list(self.torsion_store.keys()).index(key)]]

        self.update_torsions()
        self.initial_energy = deepcopy(self.mm_energies())

        # Reset the dihedral values
        self.tor_types = OrderedDict()

    def update_tor_vec(self, x):
        """Update the tor_types dict with the parameter vector."""

        x = round(x, decimals=4)

        # Update the param vector for the right torsions by slicing the vector every 4 places
        for key, val in self.tor_types.items():
            val[1] = x[key * 4:key * 4 + 4]

    def objective(self, x):
        """Return the output of the objective function."""

        # Update the parameter vector into tor_types
        self.update_tor_vec(x)

        # Update the torsions
        self.update_torsions()

        # Get the mm corresponding energy
        self.mm_energy = self.mm_energies()

        # Make sure the energies match
        assert len(self.qm_energy) == len(self.mm_energy)

        # calculate the objective

        # Adjust the mm energy to make it relative to the lowest in the scan
        self.mm_energy -= min(self.mm_energy)
        error = (self.mm_energy - self.qm_energy) ** 2

        # if using a weighting, add that here
        if self.t_weight != 'infinity':
            error *= exp(-self.qm_energy / (self.k_b * self.t_weight))

        # Find the total error
        total_error = sqrt(sum(error) / len(self.scan_coords))

        # Calculate the penalty
        pen = self.l_pen * sum((x - self.starting_params) ** 2)
        total_error += pen

        return total_error

    def run(self):
        """
        Optimise the parameters for the chosen torsions in the molecule scan_order,
        also set up a work queue to do the single point calculations if they are needed.
        """

        # Set up the first fitting
        for self.scan in self.scan_order:
            # Set the target energies first
            self.target_energy = self.energy_dict[self.scan]

            # Adjust the QM energies
            self.qm_energy = deepcopy(self.target_energy)
            self.qm_energy -= min(self.qm_energy)  # make relative to lowest energy
            self.qm_energy *= 627.509  # convert to kcal/mol

            # Get the MM coords
            self.scan_coords = self.get_coords()

            # Keep the initial coords
            self.initial_coords = deepcopy(self.scan_coords)

            # Get the initial energies
            self.initial_energies()

            # Get the torsions that will be fit and make the param vector
            self.get_torsion_params()

            # TODO start master optimiser loop
            # Start the main optimiser loop and get the final error and parameters back
            error, opt_parameters = self.scipy_optimier()

            # Push the new parameters back to the molecule parameter dictionary
            self.update_mol()

            self.plot_results(name='Iter1')

            # Do a full optimisation of the torsions to see if the energies match
            # No wavefront propagation, returns the new set of coords these become the new scan coords
            self.scan_coords = self.drive_mm()

            # Calculate the single point energies of each of the positions returned
            # Using the qm_engine, store back into the qm_energy as the new reference
            self.qm_energy = self.single_point()

            # Keep a copy of the energy before adjusting in case another loop is needed
            current_qm = deepcopy(self.qm_energy)

            # Normalise the qm energy again
            self.qm_energy -= min(self.qm_energy)  # make relative to lowest energy
            self.qm_energy *= 627.509  # convert to kcal/mol

            # Find the new error with the new coords
            validate_error = self.objective(x=opt_parameters)
            print(f'original error = {error}\nsecond scan error = {validate_error}')
            self.plot_results(name='iter2', validate=True)

            # Extend the initial energies by this new vector
            self.mm_energy = self.mm_energies()
            print(f' the new energies not corrected :\n{self.mm_energy}')
            self.initial_energy = deepcopy(append(self.initial_energy, self.mm_energy))
            print(f'all of the energies not corrected:\n{self.initial_energy}')

            # Fit again to all points
            # Put all of the coords together
            self.scan_coords = deepcopy(self.initial_coords + self.scan_coords)
            print(f'all of the scan coords:\n {len(self.scan_coords)}\n {self.scan_coords}')
            # now put all of the qm data together
            print(f'target energies not corrected:\n {self.target_energy}')
            print(f'curent qm not corrected\n {current_qm}')
            self.qm_energy = deepcopy(append(self.target_energy, current_qm))
            print(f'all of the qm energies not corrected:\n {self.qm_energy}')
            # Normalise the energy
            self.qm_energy -= min(self.qm_energy)
            self.qm_energy *= 627.509   # convert to kcal/mol

            # optimise
            error, opt_parameters = self.scipy_optimier()

            self.update_mol()

            self.plot_results(name='Final')

            # TODO optimise the single points again

            # TODO now plot the results of the scan when converged
            # TODO write out the final xml with the new parameters.
            # TODO 2D torsions using the same technique ?

    def rest_torsions(self):
        """
        Set all the torsion k values to one for every torsion in the system.

        Once an OpenMM system is created we cannot add new torsions without making a new PeriodicTorsion
        force every time.

        To get round this we have to load every k parameter into the system first; so we set every k term in the fitting
        dihedrals to 1 then reset all values to the gaff terms and update in context.
        """

        # save the molecule torsions to a dict
        self.torsion_store = deepcopy(self.molecule.PeriodicTorsionForce)

        # Set all the torsion to 1 to get them into the system
        for key in self.molecule.PeriodicTorsionForce:
            self.molecule.PeriodicTorsionForce[key] = [['1', '1', '0'], ['2', '1', '3.141592653589793'],
                                                       ['3', '1', '0'], ['4', '1', '3.141592653589793']]

        # Write out the new xml file which is read into the OpenMM system
        self.molecule.write_parameters()

        # Put the torsions back into the molecule
        self.molecule.PeriodicTorsionForce = self.torsion_store

    def get_torsion_params(self):
        """
        Get the torsions and their parameters that will scanned, work out how many different torsion types needed,
        make a vector corresponding to this size.
        """

        # Get a list of which dihedrals parameters are to be varied
        # Convert to be indexed from 0
        to_fit = [(tor[0] - 1, tor[1] - 1, tor[2] - 1, tor[3] - 1) for tor in list(self.molecule.dihedrals[self.scan])]

        # Check which ones have the same parameters and how many torsion vectors we need
        self.tor_types = OrderedDict()

        # List of torsion keys to index
        tor_key = list(self.torsion_store.keys())

        # Store the original parameter vectors to use regularisation
        self.starting_params = []

        i = 0
        while to_fit:
            # Get the current torsion
            torsion = to_fit.pop(0)

            # Get the torsions param vector used to compare to others
            # The master vector could be backwards so try one way and if keyerror try the other
            try:
                master_vector = [float(self.torsion_store[torsion][0][1]), float(self.torsion_store[torsion][1][1]),
                                 float(self.torsion_store[torsion][2][1]), float(self.torsion_store[torsion][3][1])]
            except KeyError:
                torsion = torsion[::-1]
                master_vector = [float(self.torsion_store[torsion][0][1]), float(self.torsion_store[torsion][1][1]),
                                 float(self.torsion_store[torsion][2][1]), float(self.torsion_store[torsion][3][1])]

            # Store the torsion in the starting params list
            for vn in master_vector:
                self.starting_params.append(vn)

            # Add this type to the torsion type dictionary
            self.tor_types[i] = [[torsion], master_vector, [tor_key.index(torsion)]]

            to_remove = []
            # Iterate over what is left of the list to see what other torsions are the same as the master
            for dihedral in to_fit:
                # Again, try both directions
                try:
                    vector = [float(self.torsion_store[dihedral][0][1]), float(self.torsion_store[dihedral][1][1]),
                              float(self.torsion_store[dihedral][2][1]), float(self.torsion_store[dihedral][3][1])]
                except KeyError:
                    dihedral = dihedral[::-1]
                    vector = [float(self.torsion_store[dihedral][0][1]), float(self.torsion_store[dihedral][1][1]),
                              float(self.torsion_store[dihedral][2][1]), float(self.torsion_store[dihedral][3][1])]

                # See if that vector is the same as the master vector
                if vector == master_vector:
                    self.tor_types[i][0].append(dihedral)
                    self.tor_types[i][2].append(tor_key.index(dihedral))
                    to_remove.append(dihedral)

            # Remove all of the dihedrals that have been matched
            for dihedral in to_remove:
                try:
                    to_fit.remove(dihedral)
                except ValueError:
                    to_fit.remove(dihedral[::-1])
            i += 1

        # Make the param_vector of the correct size
        self.param_vector = zeros((1, 4 * len(self.tor_types)))

    def full_scan_optimiser(self):
        """
        A steepest decent optimiser as implemented in QUBEKit-V1, which will optimise the torsion terms
        using full relaxed surface scans.
        """

        pass

    def rmsd(self):
        """
        Calculate the rmsd between the MM and QM predicted structures from the relaxed scans;
        this can be added into the penalty function.
        """

        pass

    def finite_difference(self, x):
        """Compute the gradient of changing the parameter vector using central difference scheme."""

        gradient = []
        for item in x:
            item += self.step_size / 2
            plus = self.objective(x)
            item -= self.step_size
            minus = self.objective(x)
            diff = (plus - minus) / self.step_size
            gradient.append(diff)
        return array(gradient)

    def scipy_optimiser(self):
        """The main torsion parameter optimiser that controls the optimisation method used."""

        print(f'Running SciPy {self.method} optimiser ... ')

        # Does not work in dictionary for some reason
        # TODO Try .get() ?
        if self.method == 'Nelder-Mead':
            res = minimize(self.objective, self.param_vector, method='Nelder-Mead',
                           options={'xtol': self.x_tol, 'ftol': self.error_tol, 'disp': True})

        elif self.method == 'BFGS':
            res = minimize(self.objective, self.param_vector, method='BFGS', jac=self.finite_difference,
                           options={'disp': True})

        else:
            raise NotImplementedError('The optimisation method is not implemented')

        print('SciPy optimisation complete')

        # Update the tor types dict using the optimised vector
        self.update_tor_vec(res.x)

        # return the final error and final param vector after the optimisation
        return res.fun, res.x

    def use_forcebalance(self):
        """Call force balance to do the single point energy matching."""

        pass

    def update_torsions(self):
        """Update the torsions being fitted."""

        forces = {self.simulation.system.getForce(index).__class__.__name__: self.simulation.system.getForce(index) for
                  index in range(self.simulation.system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        i = 0
        for val in self.tor_types.values():
            for j, dihedral in enumerate(val[0]):
                for v_n in range(4):
                    torsion_force.setTorsionParameters(index=v_n + (val[2][j] * 4),
                                                       particle1=dihedral[0], particle2=dihedral[1],
                                                       particle3=dihedral[2], particle4=dihedral[3],
                                                       periodicity=v_n + 1, phase=self.phases[v_n],
                                                       k=val[1][v_n])
                    i += 1
        torsion_force.updateParametersInContext(self.simulation.context)

        return self.system

    def plot_test(self, energies):
        """Plot the results of the fitting."""

        sns.set()

        # Make sure we have the same number of energy terms in the QM and MM lists
        assert len(self.qm_energy) == len(self.mm_energy)

        # Now adjust the MM energies
        # self.mm_energy -= min(self.mm_energy)
        # self.mm_energy /= 4.184 # convert from kj to kcal

        # Make the angle array
        angles = [x for x in range(-165, 195, self.qm_engine.fitting['increment'])]
        plt.plot(angles, self.qm_energy, 'o', label='QM')
        for i, scan in enumerate(energies):
            self.mm_energy = array(scan)
            self.mm_energy -= min(self.mm_energy)
            plt.plot(angles, self.mm_energy, label=f'MM{i}')
        plt.ylabel('Relative energy (kcal/mol')
        plt.xlabel('Dihedral angle$^{\circ}$')
        plt.legend()
        plt.savefig('Plot.pdf')

    def plot_results(self, name='Plot', validate=False):
        """Plot the results of the scan."""

        # sns.set()

        # Make sure we have the same number of energy terms in the QM and MM lists
        assert len(self.qm_energy) == len(self.mm_energy)

        # Adjust the MM energies
        # self.mm_energy -= min(self.mm_energy)

        # Adjust the initial MM energies
        initial_energy = self.initial_energy - min(self.initial_energy)

        # Construct the angle array
        angles = [x for x in range(-165, 195, self.qm_engine.fitting['increment'])]

        if len(self.qm_energy) > len(angles):
            points = [x for x in range(len(self.qm_energy))]
        else:
            points = None

        if points is not None:
            # Print a table of the results for multiple plots
            print(f'Geometry    QM(relative)        MM(relative)    MM_initial(relative)')
            for i in points:
                print(f'{i:4}  {self.qm_energy[i]:15.10f}     {self.mm_energy[i]:15.10f}    {initial_energy[i]:15.10f}')

            # Plot the qm and mm data
            plt.plot(points, self.qm_energy, 'o', label='QM')
            plt.plot(points, initial_energy, label='MM initial')
            plt.plot(points, self.mm_energy, label=f'MM final')

            plt.xlabel('Geometry')

        else:
            # Print a table of the results
            print(f'Angle    QM(relative)        MM(relative)    MM_initial(relative)')
            for pos, angle in enumerate(angles):
                print(f'{angle:4}  {self.qm_energy[pos]:15.10f}     {self.mm_energy[pos]:15.10f}    {initial_energy[pos]:15.10f}')

            plt.xlabel('Dihedral angle$^{\circ}$')

            # Plot the qm and mm data
            plt.plot(angles, self.qm_energy, 'o', label='QM')
            if not validate:
                plt.plot(angles, initial_energy, label='MM initial')
                plt.plot(angles, self.mm_energy, label='MM final')

            else:
                plt.plot(angles, self.mm_energy, label='MM validate')

        # Label the graph and save the pdf
        plt.ylabel('Relative energy (kcal/mol)')
        plt.legend(loc=1)
        plt.savefig(f'{name}.pdf')
        plt.clf()

    def make_constraints(self):
        """Write a constraint file used by geometric during optimizations."""

        with open('constraints.txt', 'w+')as constraint:
            constraint.write(
                f'$scan\ndihedral {self.molecule.dihedrals[self.scan][0][0]} {self.molecule.dihedrals[self.scan][0][1]}'
                f' {self.molecule.dihedrals[self.scan][0][2]} {self.molecule.dihedrals[self.scan][0][3]} -165.0 180 23\n')

    def write_dihedrals(self):
        """Write out the torsion drive dihedral file for the current self.scan."""

        with open('dihedrals.txt', 'w+') as out:
            out.write('# dihedral definition by atom indices starting from 0\n# i     j     k     l\n')
            mol_di = self.molecule.dihedrals[self.scan][0]
            out.write(f'  {mol_di[0]}     {mol_di[1]}     {mol_di[2]}     {mol_di[3]}\n')

    def drive_mm(self):
        """Drive the torsion again using MM to get new structures."""

        # Create a temporary working directory to call torsion drive from
        # Write an xml file with the new parameters

        # Move into a temporary folder
        # Turned off for testing
        # with TemporaryDirectory() as temp:
        temp = 'tester'
        mkdir(temp)
        chdir(temp)

        # Write out a pdb file of the qm optimised geometry
        self.molecule.write_pdb(name='openmm')
        # Also need an xml file for the molecule to use in geometric
        self.molecule.write_parameters(name='input')
        # openmm.pdb and input.xml are the expected names for geometric
        print('Making the constraint file')
        self.make_constraints()
        print('making dihedrals file')
        self.write_dihedrals()
        print('running torsion drive ...')
        with open('log.txt', 'w+')as log:
            sub_call('torsiondrive-launch -e openmm openmm.pdb dihedrals.txt', shell=True, stdout=log)
            # sub_call('geometric-optimize --openmm openmm.pdb constraints.txt', shell=True, stdout=log)
        print('gathering the new positions ...')

        # return the new positions
        return self.get_coords()

    def single_point(self):
        """Take set of coordinates of a molecule and do a single point calculation; returns an array of the energies."""

        sp_energy = []
        # for each coordinate in the system we need to write a qm input file and get the single point energy
        # TODO add progress bar (tqdm?)
        for i, x in enumerate(self.scan_coords):
            mkdir(f'SP_{i}')
            chdir(f'SP_{i}')
            print(f'Doing single point calculations on new structures ... {i + 1}/{len(self.scan_coords)}')
            # now we need to change the positions of the molecule in the molecule array
            for y, coord in enumerate(x):
                for z, pos in enumerate(coord):
                    self.qm_engine.engine_mol.molecule[y][z + 1] = pos * 10  # convert from nanometers in openmm to A in QM

            # Write the new coordinate file and run the calculation
            self.qm_engine.generate_input(energy=True)

            # Extract the energy and save to the array
            sp_energy.append(self.qm_engine.get_energy())

            # Move back to the base directory
            chdir('../')

        # return the array of the new single point energies
        return array(sp_energy)

    def update_mol(self):
        """When the optimisation is complete update the PeriodicTorsionForce parameters in the molecule."""

        for val in self.tor_types.values():
            for dihedral in val[0]:
                for vn in range(4):
                    self.molecule.PeriodicTorsionForce[dihedral][vn][1] = str(val[1][vn])

    def opls_lj(self, excep_pairs=None, normal_pairs=None):
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
            l_j_set[index] = (sigma, epsilon)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)

        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED FORCE
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                sig14 = sqrt(l_j_set[p1][0] * l_j_set[p2][0])
                # TODO eps14 not used
                eps14 = sqrt(l_j_set[p1][1] * l_j_set[p2][1])
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
            # Old method which needs updating
            if excep_pairs:
                for x in range(len(excep_pairs)):  # scale 14 interactions
                    if p1 == excep_pairs[x, 0] and p2 == excep_pairs[x, 1] or p2 == excep_pairs[x, 0] and p1 == excep_pairs[x, 1]:
                        charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
                        charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
                        q = charge1 * charge2 * 0.5
                        sig14 = sqrt(sigma1 * sigma2) * 0.5
                        eps = sqrt(epsilon1 * epsilon2) * 0.5
                        nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

            if normal_pairs:
                for x in range(len(normal_pairs)):
                    if p1 == normal_pairs[x, 0] and p2 == normal_pairs[x, 1] or p2 == normal_pairs[x, 0] and p1 == normal_pairs[x, 1]:
                        charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
                        charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
                        q = charge1 * charge2
                        sig14 = sqrt(sigma1 * sigma2)
                        eps = sqrt(epsilon1 * epsilon2)
                        nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

        return self.system
