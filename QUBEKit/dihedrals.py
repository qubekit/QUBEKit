#!/usr/bin/env python

from subprocess import call as sub_call
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from numpy import array, zeros, sqrt, sum, exp, round, append
from math import pi
from collections import OrderedDict
from copy import deepcopy
from os import chdir, mkdir
from tempfile import TemporaryDirectory
from subprocess import call
from QUBEKit.decorators import timer_func, for_all_methods



class TorsionScan:
    """This class will take a QUBEKit molecule object and perform a torsiondrive QM (and MM if True) energy scan
    for each selected dihedral.
    """

    def __init__(self, molecule, qmengine, mmengine='openmm', native_opt=False,
                 verbose=False):
        self.QMengine = qmengine
        self.MMengine = mmengine
        self.constraints = None
        self.grid_space = qmengine.fitting['increment']
        self.native_opt = native_opt
        self.verbose = verbose
        self.scan_mol = molecule
        self.cmd = {}
        self.find_scan_order()
        self.torsion_cmd()

    def find_scan_order(self):
        """Function takes the molecule and displays the rotatable central bonds,
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
            # get the rotatable dihedrals from the molecule
            rotatable = list(self.scan_mol.rotatable)
            print('Please select the central bonds round which you wish to scan in the order to be scanned')
            print('Torsion number   Central-Bond   Representative Dihedral')
            for i, bond in enumerate(rotatable):
                print(f'  {i+1}                    {bond[0]}-{bond[1]}             '
                        f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][0]-1]}-'
                        f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][1]-1]}-'
                        f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][2]-1]}-'
                        f'{self.scan_mol.atom_names[self.scan_mol.dihedrals[bond][0][3]-1]}')

            scans = list(input('>'))  # Enter as a space separated list
            scans[:] = [scan for scan in scans if scan != ' ']  # remove all spaces from the scan list
            print(scans)

            scan_order = []
            # now add the rotatable dihedral keys to an array
            for scan in scans:
                scan_order.append(rotatable[int(scan)-1])
            self.scan_mol.scan_order = scan_order

            return self.scan_mol

    def qm_scan_input(self, scan, run=False):
        """Function takes the rotatable dihedrals requested and writes a scan input file for torsiondrive."""

        with open('dihedrals.txt', 'w+') as out:

            out.write('# dihedral definition by atom indices starting from 0\n# i     j     k     l\n')
            out.write('  {}     {}     {}     {}\n'.format(self.scan_mol.dihedrals[scan][0][0], self.scan_mol.dihedrals[scan][0][1], self.scan_mol.dihedrals[scan][0][2], self.scan_mol.dihedrals[scan][0][3]))
        # TODO need to add PSI4 redundant mode selector

        if self.native_opt:
            self.QMengine.generate_input(optimize=True, threads=True)

        else:
            self.QMengine.geo_gradient(run=False, threads=True)

    def torsion_cmd(self):
        """Function generates a command strings to run torsiondrive based on the input commands for QM and MM."""

        # add the first basic command elements for QM
        cmd_qm = f'torsiondrive-launch {self.scan_mol.name}.{self.QMengine.__class__.__name__.lower()}in dihedrals.txt '
        if self.grid_space:
            cmd_qm += f'-g {self.grid_space} '
        if self.QMengine:
            cmd_qm += f'-e {self.QMengine.__class__.__name__.lower()} '

        if self.native_opt:
            cmd_qm += '--native_opt '
        if self.verbose:
            cmd_qm += '-v '

        self.cmd = cmd_qm
        return self.cmd

    def get_energy(self, scan):
        """Function will extract an array of energies from the scan results
        and store it back into the molecule in a dictionary using the scan order as keys.
        """
        from numpy import array
        with open('scan.xyz', 'r') as scan_file:
            scan_energy = []
            for line in scan_file:
                if 'Energy ' in line:
                    scan_energy.append(float(line.split()[3]))

            self.scan_mol.QM_scan_energy[scan] = array(scan_energy)

            return self.scan_mol

    def start_scan(self):
        """Function makes a folder and writes a new a dihedral input file for each scan."""
        # TODO put all the scans in a work queue so they can be performed in parallel

        from os import mkdir, chdir

        for scan in self.scan_mol.scan_order:
            try:
                mkdir(f'SCAN_{scan}')

            except:
                raise Exception(f'Cannot create SCAN_{scan} dir.')

            chdir(f'SCAN_{scan}')
            mkdir('QM')
            chdir('QM')
            # now make the scan input files
            self.qm_scan_input(scan)
            sub_call(self.cmd, shell=True)
            self.get_energy(scan)
            chdir('../')


# @for_all_methods(timer_func)
class TorsionOptimizer:
    """Torsion optimizer class used to optimize dihedral parameters with a range of methods"""

    def __init__(self, molecule, qmengine, config_dict, wieght_mm=True, opls=True, use_force=False, step_size=0.002 , error_tol=1e-5, xtol=1e-4 , method='BFGS'):
        self.qm, self.fitting, self.descriptions = config_dict[1], config_dict[2], config_dict[3]
        self.l_pen = self.fitting['l_pen']
        self.t_weight = self.fitting['t_weight']
        self.molecule = molecule
        self.QMengine = qmengine
        self.OPLS = opls
        self.weight_MM = wieght_mm
        self.step_size = step_size
        self.methods = {'NM': 'Nelder-Mead',  # scipy method
                        'BFGS': 'BFGS',  # BFGS method in scipy with custom step size
                        }
        self.method = self.methods[method]
        self.error_tol = error_tol
        self.xtol = xtol
        self.energy_dict = molecule.QM_scan_energy
        self.use_Force = use_force
        self.MM_energy = []
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
        self.K_b = 0.001987
        self.tor_types = OrderedDict()
        self.phases = [0, pi, 0, pi]
        self.rest_torsions()
        self.openmm_system()

    def mm_energies(self):
        """Evaluate the MM energies of the QM structures."""

        self.MM_energy = []
        for position in self.scan_coords:
            # update the positions of the system
            self.simulation.context.setPositions(position)

            # Then get the energy from the new state
            state = self.simulation.context.getState(getEnergy=True, getForces=self.use_Force)
            # print(f'{float(str(state.getPotentialEnergy())[:-6])/4.184} kcal/mol')
            self.MM_energy.append(float(str(state.getPotentialEnergy())[:-6])/4.184)  # convert from kj to kcal
        return array(self.MM_energy)
        # get forces from the system
        # open_grad = state.getForces()

    def get_coords(self):
        """Read the torsion drive output file to get all of the coords in a format that can be passed to openmm
        so we can update positions in context without reloading the molecule."""

        scan_coords = []
        # open the torsion drive data file read all the scan coordinates
        with open('qdata.txt', 'r') as data:
            for line in data.readlines():
                if 'COORDS' in line:
                    # get the coords into a single array
                    cords = [float(x)/10 for x in line.split()[1:]]
                    # convert to a list of tuples
                    tups = []
                    for i in range(0, len(cords), 3):
                        tups.append((cords[i], cords[i+1], cords[i+2]))
                    scan_coords.append(tups)

        return scan_coords

    def openmm_system(self):
        """Initialise the OpenMM system we will use to evaluate the energies."""

        # load the initial coords into the system and initialise

        pdb = app.PDBFile(self.molecule.filename)
        forcefield = app.ForceField(f'{self.molecule.name}.xml')
        modeller = app.Modeller(pdb.topology, pdb.positions)  # set the intial positions from the pdb
        self.system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
        if self.OPLS:
            self.opls_lj()
        temperature = 298.15 * unit.kelvin
        integrator = mm.LangevinIntegrator(temperature, 5 / unit.picoseconds, 0.001 * unit.picoseconds)
        self.simulation = app.Simulation(modeller.topology, self.system, integrator)
        self.simulation.context.setPositions(modeller.positions)

    def initial_energys(self):
        """Calculate the initial energies using the input xml."""

        # First we have to reset all of the Periodic torsion terms back to their initial values
        i = 0
        for key in self.torsion_store:
            self.tor_types[i] = [[key], [float(self.torsion_store[key][0][1]), float(self.torsion_store[key][1][1]), float(self.torsion_store[key][2][1]), float(self.torsion_store[key][3][1])], [list(self.torsion_store.keys()).index(key)]]
            i += 1
        # print(self.tor_types)

        self.update_torsions()

        self.initial_energy = deepcopy(self.mm_energies())

        # now reset the dihedral values
        self.tor_types = OrderedDict()

    def update_tor_vec(self, x):
        """Update the tor_types dict with the parameter vector."""

        # round the numbers to 4 d.p
        x = round(x, decimals=4)
        # print(f'current param vector {x}')

        # up date the param vector for the right torsions by slicing the vector every 4 places
        for key in self.tor_types.keys():
            self.tor_types[key][1] = x[key*4:key*4+4]

    def objective(self, x):
        """Return the output of the objective function."""

        # first update the parameter vector into tor_types
        self.update_tor_vec(x)

        # now update the torsions
        self.update_torsions()

        # get the mm corresponding energy
        self.MM_energy = self.mm_energies()

        # make sure the amount of energies match
        assert len(self.qm_energy) == len(self.MM_energy)

        # calculate the objective

        # first we adjust the mm energy to make it relative to the lowest in the scan
        self.MM_energy -= min(self.MM_energy)
        ERRA = (self.MM_energy - self.qm_energy) ** 2

        # if we are using a weighting add that here
        if self.t_weight != 'infinity':
            ERRA *= exp(-self.qm_energy/(self.K_b * self.t_weight))

        # now find the total error
        ERRS = sqrt(sum(ERRA) / (len(self.scan_coords)))

        # calculate the penalty
        pen = self.l_pen * sum((x - self.starting_params)**2)
        ERRS += pen

        return ERRS

    def run(self):
        """Optimize the parameters for the chosen torsions in the molecule scan_order,
        also set up a work queue to do the single point calculations if they are needed."""

        # set up the first fitting
        for self.scan in self.scan_order:
            # set the target energies first
            self.target_energy = self.energy_dict[self.scan]

            # Adjust the QM energies
            self.qm_energy = deepcopy(self.target_energy)
            self.qm_energy -= min(self.qm_energy) # make relative to lowest energy
            self.qm_energy *= 627.509 # convert to kcal/mol

            # Get the MM coords
            self.scan_coords = self.get_coords()

            # keep the intial coords
            self.initial_coords = deepcopy(self.scan_coords)

            # get the initial energies
            self.initial_energys()

            # now get the torsions to be fit
            # get the torsions that will be fit and make the parm vector
            self.get_torsion_params()

            #TODO start master optimizer loop
            # now start the main optimizer loop and get the final error and parameters back
            error, opt_parameters = self.scipy_optimier()

            # first push the new parameters back to the molecules parameter dictionary
            self.update_mol()

            self.plot_results(name='Iter1')

            # now do a full optimization of the torsions to see if the energies match
            # no wavefront propagation, returns the new set of coords these become the new scan coords
            self.scan_coords = self.drive_mm()

            # now calculate the single point energies of each of the positions returned
            # using the QMengine, store back into the qm_energy as the new reference
            self.qm_energy = self.single_point()

            # keep a copy of the energy before adjusting incase we need another loop
            current_qm = deepcopy(self.qm_energy)

            # normalize the qm energy again
            self.qm_energy -= min(self.qm_energy)  # make relative to lowest energy
            self.qm_energy *= 627.509  # convert to kcal/mol

            # find the new error with the new coords
            validate_error = self.objective(x=opt_parameters)
            print(f'original error = {error}\nsecond scan error = {validate_error}')
            self.plot_results(name='iter2', validate=True)

            # we should also extend the initial energies by this new vector
            self.MM_energy = self.mm_energies()
            print(f' the new energies not corrected :\n{self.MM_energy}')
            self.initial_energy = deepcopy(append(self.initial_energy, self.MM_energy))
            print(f'all of the energies not corrected:\n{self.initial_energy}')


            # now fit again to all points
            # put all of the coords together
            self.scan_coords = deepcopy(self.initial_coords + self.scan_coords)
            print(f'all of the scan coords:\n {len(self.scan_coords)}\n {self.scan_coords}')
            # now put all of the qm data together
            print(f'target energies not corrected:\n {self.target_energy}')
            print(f'curent qm not corrected\n {current_qm}')
            self.qm_energy = deepcopy(append(self.target_energy, current_qm))
            print(f'all of the qm energies not corrected:\n {self.qm_energy}')
            # now normalize the energy
            self.qm_energy -= min(self.qm_energy)
            self.qm_energy *= 627.509  # convert to kcal/mol

            # optimize
            error, opt_parameters = self.scipy_optimier()

            self.update_mol()

            self.plot_results(name='Final')

            #TODO optimize the single points again

            #TODO now plot the results of the scan when converged
            #TODO write out the final xml with the new parameters.
            #TODO 2D torsions using the same technique ?




    def rest_torsions(self):
        """Set all the torsion k values to one for every torsion in the system.

        Info
        ---------------
        Once an OpenMM system is created we can not add new torsions with out making a new PeriodicTorsion
        force every time.

        To get round this we have to load every k parameter into the system first, so we set every k term in the fitting
        dihedrals to 1 then reset all values to the gaff terms and update in context.
        ----------------"""

        # save the molecule torsions to a dict
        self.torsion_store = deepcopy(self.molecule.PeriodicTorsionForce)

        # Now set all the torsion to 1 to get them into the system

        for key in self.molecule.PeriodicTorsionForce:
            self.molecule.PeriodicTorsionForce[key] = [['1', '1', '0'], ['2', '1', '3.141592653589793'], ['3', '1', '0'], ['4', '1', '3.141592653589793']]

        # Now write out the new xml file which is read into the OpenMM system
        self.molecule.write_parameters()

        # now put the torsions back into the molecule
        self.molecule.PeriodicTorsionForce = self.torsion_store

    def get_torsion_params(self):
        """Get the torsions and their parameters that will scanned, work out how many different torsion types needed,
        make a vector corresponding to this size."""

        # first get a list of which dihedrals parameters are to be varied
        # convert to be indexed from 0
        self.to_fit = [(tor[0]-1, tor[1]-1, tor[2]-1, tor[3]-1) for tor in list(self.molecule.dihedrals[self.scan])]
        print(self.to_fit)
        # Now check which ones have the same parameters and how many torsion vectors we need
        self.tor_types = OrderedDict()

        # list of torsion keys to index
        self.tor_key = list(self.torsion_store.keys())

        # we need to store the original parameter vectors so we can use regularization
        self.starting_params = []

        i = 0
        while self.to_fit:
            # get the current torsion
            torsion = self.to_fit.pop(0)

            # get the torsions param vector used to compare to others
            # the master vector could be in the wrong order so we need to try on way and if keyerror try the other
            try:
                master_vector = [float(self.torsion_store[torsion][0][1]), float(self.torsion_store[torsion][1][1]), float(self.torsion_store[torsion][2][1]), float(self.torsion_store[torsion][3][1])]
            except KeyError:
                torsion = torsion[::-1]
                master_vector = [float(self.torsion_store[torsion][0][1]), float(self.torsion_store[torsion][1][1]), float(self.torsion_store[torsion][2][1]), float(self.torsion_store[torsion][3][1])]

            # also store the torsion in the starting params list
            for vn in master_vector:
                self.starting_params.append(vn)
            print(self.starting_params)

            # add this type to the torsion type dictionary
            self.tor_types[i] = [[torsion], master_vector, [self.tor_key.index(torsion)]]

            to_remove = []
            # now iterate over what is left of the list to see what other torsions are the same as the master
            for dihedral in self.to_fit:
                # we may have the torsion particles in a different order so try both ways
                try:
                    vector = [float(self.torsion_store[dihedral][0][1]), float(self.torsion_store[dihedral][1][1]),
                              float(self.torsion_store[dihedral][2][1]), float(self.torsion_store[dihedral][3][1])]
                except KeyError:
                    dihedral = dihedral[::-1]
                    vector = [float(self.torsion_store[dihedral][0][1]), float(self.torsion_store[dihedral][1][1]),
                              float(self.torsion_store[dihedral][2][1]), float(self.torsion_store[dihedral][3][1])]

                # now see if that vector is the same as the master vector
                if vector == master_vector:
                    self.tor_types[i][0].append(dihedral)
                    self.tor_types[i][2].append(self.tor_key.index(dihedral))
                    to_remove.append(dihedral)

            # now remove all of the dihedrals that have been matched
            for dihedral in to_remove:
                try:
                    self.to_fit.remove(dihedral)
                except ValueError:
                    self.to_fit.remove(dihedral[::-1])
            i += 1

        # now make the param_vector of the correct size
        self.param_vector = zeros((1, len(list(self.tor_types.keys()))*4))

    def full_scan_optimizer(self):
        """A steepest decent optimizer as implimented in QUBEKit-V1, that will optimize the torsion terms
         using full relaxed surface scans."""
        pass

    def rmsd(self):
        """Calculate the rmsd between the MM and QM predicted structures from the relaxed scans
        this can be added into the penalty function."""

        pass

    def finite_difference(self, x):
        """Compute the gradient of changing the parameter vector using central difference scheme."""

        gradient = []
        for i in range(len(x)):
            x[i] += self.step_size/2
            plus = self.objective(x)
            x[i] -= self.step_size
            minus = self.objective(x)
            diff = (plus-minus)/self.step_size
            gradient.append(diff)
        # print(f'gradient: {gradient}')
        return array(gradient)

    def scipy_optimier(self):
        """The main torsion parameter optimizer that controls the optimization method used."""

        from scipy.optimize import minimize

        # select the scipy minimize method from the ones available

        # run the optimizer
        print(f'Running scipy {self.method} optimizer ... ')
        # does not work in dictionary for some reason
        if self.method == 'Nelder-Mead':
            res = minimize(self.objective, self.param_vector, method='Nelder-Mead', options={'xtol': self.xtol, 'ftol': self.error_tol, 'disp': True})

        elif self.method == 'BFGS':
            res =  minimize(self.objective, self.param_vector, method='BFGS', jac=self.finite_difference, options = {'disp': True})

        else:
            raise NotImplementedError('The optimisation method is not implemented')

        print('Scipy optimization Done')

        # now update the tor types dict using the optimized vector
        self.update_tor_vec(res.x)

        # return the final error and final param vector after the optimization
        return res.fun, res.x


    def use_forcebalance(self):
        """Call force balance to do the single point energy matching."""
        pass

    def update_torsions(self):
        """Update the torsions being fitted."""

        # print('Updating dihedrals')

        forces = {self.simulation.system.getForce(index).__class__.__name__: self.simulation.system.getForce(index) for index in
                  range(self.simulation.system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']
        i = 0
        for key in self.tor_types.keys():
            # print(self.tor_types[key])
            for j, dihedral in enumerate(self.tor_types[key][0]):
                for v_n in range(4):
                    torsion_force.setTorsionParameters(index=v_n+(self.tor_types[key][2][j]*4), particle1=dihedral[0], particle2=dihedral[1], particle3=dihedral[2], particle4=dihedral[3],
                                               periodicity=v_n+1, phase=self.phases[v_n], k=self.tor_types[key][1][v_n])
                    i += 1
        # for dihedral in range(torsion_force.getNumTorsions()):
        #     print(torsion_force.getTorsionParameters(dihedral))
        torsion_force.updateParametersInContext(self.simulation.context)

        return self.system

    def plot_test(self, energies):
        """Plot the results of the fitting."""

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        # Make sure we have the same amount of energy terms in the QM and mm lists
        assert len(self.qm_energy) == len(self.MM_energy)

        # Now adjust the MM energies
        # self.MM_energy -= min(self.MM_energy)
        # self.MM_energy /= 4.184 # convert from kj to kcal

        # make the angle array
        angles = [x for x in range(-165, 195, self.QMengine.fitting['increment'])]
        plt.plot(angles, self.qm_energy, 'o', label='QM')
        for i, scan in enumerate(energies):
            self.MM_energy = array(scan)
            self.MM_energy -= min(self.MM_energy)
            plt.plot(angles, self.MM_energy,  label=f'MM{i}')
        plt.ylabel('Relative energy (kcal/mol')
        plt.xlabel('Dihedral angle$^{\circ}$')
        plt.legend()
        plt.savefig('Plot.pdf')

    def plot_results(self, name='Plot', validate=False):
        """Plot the results of the scan."""

        import matplotlib.pyplot as plt
        import seaborn as sns
        # sns.set()

        # Make sure we have the same amount of energy terms in the QM and mm lists
        assert len(self.qm_energy) == len(self.MM_energy)

        # Now adjust the MM energies
        # self.MM_energy -= min(self.MM_energy)

        # Now adjust the initial MM energies
        initial_energy = self.initial_energy - min(self.initial_energy)

        # make the angle array
        angles = [x for x in range(-165, 195, self.QMengine.fitting['increment'])]

        if len(self.qm_energy) > len(angles):
            points = [x for x in range(len(self.qm_energy))]
        else:
            points = None

        if points:
            # print a nice table of all of the results together for multipule plots
            print(f'Geometry    QM(relative)        MM(relative)    MM_initial(relative)')
            for i in points:
                print(f'{i:4}  {self.qm_energy[i]:15.10f}     {self.MM_energy[i]:15.10f}    {initial_energy[i]:15.10f}')

            # Now plot the QM and MM data
            plt.plot(points, self.qm_energy, 'o', label='QM')
            plt.plot(points, initial_energy, label='MM initial')
            plt.plot(points, self.MM_energy, label=f'MM final')

            plt.xlabel('Geometry')

        else:
            # print a nice table of all of the results together
            print(f'Angle    QM(relative)        MM(relative)    MM_initial(relative)')
            for i, angle in enumerate(angles):
                print(f'{angle:4}  {self.qm_energy[i]:15.10f}     {self.MM_energy[i]:15.10f}    {initial_energy[i]:15.10f}')

            plt.xlabel('Dihedral angle$^{\circ}$')

            if not validate:
                # Now plot the QM and MM data
                plt.plot(angles, self.qm_energy, 'o', label='QM')
                plt.plot(angles, initial_energy, label='MM initial')
                plt.plot(angles, self.MM_energy, label=f'MM final')

            elif validate:
                # Now plot the QM and MM data
                plt.plot(angles, self.qm_energy, 'o', label='QM')
                plt.plot(angles, self.MM_energy, label=f'MM validate')


        # label the graph and save the pdf
        plt.ylabel('Relative energy (kcal/mol)')
        plt.legend(loc=1)
        plt.savefig(f'{name}.pdf')
        plt.clf()

        print('Torsion graph made!')

    def make_constraints(self):
        """Write a constraint file used by geometric during optimizations."""

        with open('constraints.txt', 'w+')as constraint:
            constraint.write(f'$scan\ndihedral {self.molecule.dihedrals[self.scan][0][0]} {self.molecule.dihedrals[self.scan][0][1]} {self.molecule.dihedrals[self.scan][0][2]} {self.molecule.dihedrals[self.scan][0][3]} -165.0 180 23\n')

    def write_dihedrals(self):
        """Write out the torsion drive dihedral file for the current self.scan."""

        with open('dihedrals.txt', 'w+') as out:
            out.write('# dihedral definition by atom indices starting from 0\n# i     j     k     l\n')
            out.write('  {}     {}     {}     {}\n'.format(self.molecule.dihedrals[self.scan][0][0],
                                                           self.molecule.dihedrals[self.scan][0][1],
                                                           self.molecule.dihedrals[self.scan][0][2],
                                                           self.molecule.dihedrals[self.scan][0][3]))

    def drive_mm(self):
        """Drive the torsion again using MM to get new structures."""

        # create a tempary working directory where we can call torsion drive to scan the torsion
        # now we need to write out a new xml file with the new parameters in

        # move into a tempary folder
        # truned of for testing
        #with TemporaryDirectory() as temp:
        temp = 'tester'
        mkdir(temp)

        chdir(temp)

        # now write out a pdb file of the QM optimized geometry
        self.molecule.write_pdb(name='openmm')
        # we also need an xml file for the molecule to use in geometric
        self.molecule.write_parameters(name='input')
        # openmm.pdb and input.xml are the expected names for geometric
        print('Making the constraint file')
        self.make_constraints()
        print('making dihedrals file')
        self.write_dihedrals()
        print('running torsion drive ....')
        with open('log.txt', 'w+')as log:
            call('torsiondrive-launch -e openmm openmm.pdb dihedrals.txt', shell=True, stdout=log)
            # call('geometric-optimize --openmm openmm.pdb constraints.txt', shell=True, stdout=log)
        print('gathering the new positions ...')

        # get the new positions
        new_coords = self.get_coords()
        # return the new parameter coords
        return new_coords

    def single_point(self):
        """Take set of coordinates of a molecule and do a single point calculations returns a array of the energies."""

        SP_energy = []
        # for each coordinate in the system we need to write a QMinput file and get the single point energy
        #TODO add progress bar
        for i, x in enumerate(self.scan_coords):
            mkdir(f'SP_{i}')
            chdir(f'SP_{i}')
            print(f'Doing single point calculations on new structures ... {i+1}/{len(self.scan_coords)}')
            # now we need to change the positions of the molecule in the molecule array
            for y, coord in enumerate(x):
                for z, pos in enumerate(coord):
                    self.QMengine.engine_mol.molecule[y][z+1] = pos*10  # convert from nanometers in openmm to A in QM

            # now try and write the new coordinate file and run the calculation
            self.QMengine.generate_input(energy=True)

            # extract the energy and save to the array
            SP_energy.append(self.QMengine.get_energy())

            # move back to the base directory
            chdir('../')

        # return the array of the new single point energies
        return array(SP_energy)

    def update_mol(self):
        """When the optimization is complete update the PeriodicTorsionForce parameters in the molecule."""

        for key in self.tor_types.keys():
            for dihedral in self.tor_types[key][0]:
                for vn in range(4):
                    self.molecule.PeriodicTorsionForce[dihedral][vn][1] = str(self.tor_types[key][1][vn])


    def opls_lj(self, excep_pairs=None, normal_pairs=None):
        """This function changes the standard OpenMM combination rules to use OPLS, execp and normal pairs are only
        required if their are virtual sites in the molecule."""

        # get system information from the openmm system
        forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in
                  range(self.system.getNumForces())}
        # use the nondonded_force tp get the same rules
        nonbonded_force = forces['NonbondedForce']
        lorentz = mm.CustomNonbondedForce(
            'epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)*4.0')
        lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
        lorentz.addPerParticleParameter('sigma')
        lorentz.addPerParticleParameter('epsilon')
        lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
        self.system.addForce(lorentz)
        LJset = {}
        # Now for each particle calculate the combination list again
        for index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
            # print(nonbonded_force.getParticleParameters(index))
            LJset[index] = (sigma, epsilon)
            lorentz.addParticle([sigma, epsilon])
            nonbonded_force.setParticleParameters(
                index, charge, sigma, epsilon * 0)
        for i in range(nonbonded_force.getNumExceptions()):
            (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
            # ALL THE 12,13 and 14 interactions are EXCLUDED FROM CUSTOM NONBONDED
            # FORCE
            lorentz.addExclusion(p1, p2)
            if eps._value != 0.0:
                sig14 = sqrt(LJset[p1][0] * LJset[p2][0])
                eps14 = sqrt(LJset[p1][1] * LJset[p2][1])
                nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            # If there is a virtual site in the molecule we have to change the exceptions and pairs lists
            # old method that needs updating
            if excep_pairs:
                for x in range(len(excep_pairs)):  # scale 14 interactions
                    if p1 == excep_pairs[x, 0] and p2 == excep_pairs[x, 1] or p2 == excep_pairs[x, 0] and p1 == excep_pairs[
                        x, 1]:
                        charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
                        charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
                        q = charge1 * charge2 * 0.5
                        # print('charge %s'%q)
                        sig14 = sqrt(sigma1 * sigma2) * 0.5
                        eps = sqrt(epsilon1 * epsilon2) * 0.5
                        nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)
            if normal_pairs:
                for x in range(len(normal_pairs)):
                    if p1 == normal_pairs[x, 0] and p2 == normal_pairs[x, 1] or p2 == normal_pairs[x, 0] and p1 == \
                            normal_pairs[
                                x, 1]:
                        charge1, sigma1, epsilon1 = nonbonded_force.getParticleParameters(p1)
                        charge2, sigma2, epsilon2 = nonbonded_force.getParticleParameters(p2)
                        q = charge1 * charge2
                        # print(q)
                        sig14 = sqrt(sigma1 * sigma2)
                        eps = sqrt(epsilon1 * epsilon2)
                        nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps)

        return self.system
