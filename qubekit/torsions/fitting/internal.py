import os
import subprocess as sp
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from shutil import rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import linregress

from qubekit.engines import OpenMM
from qubekit.molecules.ligand import Ligand
from qubekit.utils import constants
from qubekit.utils.file_handling import make_and_change_into

matplotlib.use("Agg")  # Fix for clusters?


class TorsionOptimiser:
    """
    Torsion optimiser class used to optimise dihedral parameters with a range of methods

    inputs
    ---------
    # Configurations
    l_pen
    t_weight
    weight_mm:              Weight the low energy parts of the surface (not sure if it works)
    step_size:              The scipy displacement step size
    methods
    method
    error_tol
    x_tol:
    use_force:              Match the forces as well as the energies (not available yet)
    abs_bounds
    refinement:             The stage two refinement methods

    # QUBEKit Internals
    molecule
    qm_engine

    # TorsionOptimiser starting parameters
    scans_dict              QM scan energies {(scan): [array of qm energies]}
    mm_energy               numpy array of the current mm energies
    initial_energy          numpy array of the fitting iteration initial parameter energies
    starting_energy         numpy array of the starting parameter energies
    scan_order              list of the scan keys in the order to be fit
    scan_coords             list of molecule geometries in OpenMM format list[tuple] [(x, y, z)]
    starting_params         list of the dihedral starting parameters
    energy_store_qm         list of all of the qm energies collected in the same order as the scan coords
    coords_store            list of all of the coordinates sampled in the fitting
    initial_coords          the qm optimised geometries
    tor_types               important! stores the torsion indices in the OpenMM system and groups torsions
    target_energy           list of the qm optimised energies
    qm_energy               the current qm energy numpy array
    scan                    the current scan key that is being fit
    param_vector            numpy array of the parameters being fit, this is a flat array even with multiple torsions
    torsion_store           this dictionary is a copy of the molecules periodic torsion force dict
    index_dict              used to work out the index of the torsions in the OpenMM system
    qm_local                the location of the QM torsiondrive
    """

    def __init__(
        self,
        molecule: Ligand,
        weight_mm=True,
        step_size=0.02,
        error_tol=1e-5,
        x_tol=1e-5,
    ):

        self.molecule = molecule
        self.qm_engine = {"psi4": PSI4, "g09": Gaussian, "g16": Gaussian}.get(
            self.molecule.bonds_engine
        )(molecule)

        # Configurations
        self.weight_mm = weight_mm
        self.step_size = step_size
        self.error_tol = error_tol
        self.x_tol = x_tol

        self.l_pen = self.molecule.l_pen
        self.t_weight = self.molecule.t_weight
        self.abs_bounds = molecule.tor_limit
        self.refinement = molecule.refinement_method
        # Scipy optimisation methods also hybryd methods that use one method for the first fit then switch
        # eg (GA_BFGS, GA_NM, NM_GA, BFGS_GA)
        self.methods = {"NM": "Nelder-Mead", "BFGS": "BFGS", "GA": "Genetic"}
        self.method = self.methods.get(self.molecule.opt_method, None)

        # TorsionOptimiser starting parameters
        self.scans_dict = deepcopy(molecule.qm_scans)
        self.mm_energy = None
        self.initial_energy = None
        self.starting_energy = None
        self.scan_coords = None
        self.starting_params = None
        self.energy_store_qm = []
        self.coords_store = []
        self.initial_coords = []
        self.tor_types = OrderedDict()
        self.target_energy = None
        self.qm_energy = None
        self.scan = None
        self.param_vector = None
        self.torsion_store = None
        self.index_dict = {}
        self.qm_local = None
        self.rmsd_atoms = []
        self.optimiser_log = None

        # Convert the optimised qm coords to OpenMM format
        self.opt_coords = self.molecule.openmm_coordinates()

        # constants
        self.k_b = constants.KB_KCAL_P_MOL_K
        self.phases = [0, constants.PI, 0, constants.PI]
        self.home = os.getcwd()

        # start the OpenMM system
        self.load_torsions()
        # Now start the OpenMM engine
        self.open_mm = OpenMM(self.molecule)

    def mm_energies(self):
        """
        Evaluate the MM energies of the geometries stored in scan_coords.
        :return: A numpy array of the energies for easy normalisation.
        """

        return np.array(
            [self.open_mm.get_energy(position) for position in self.scan_coords]
        )

    def reset_torsions(self):
        """Reset all torsion values to their initial and create the torsion index dictionary."""

        # first we need to work out the index order the torsions are in while inside the OpenMM system
        # this order is different from the xml order
        forces = {
            self.open_mm.simulation.system.getForce(
                index
            ).__class__.__name__: self.open_mm.simulation.system.getForce(index)
            for index in range(self.open_mm.simulation.system.getNumForces())
        }
        torsion_force = forces["PeriodicTorsionForce"]

        for i in range(torsion_force.getNumTorsions()):
            # torsion, periodicity, phase, k
            *torsion, _, _, _ = torsion_force.getTorsionParameters(i)
            torsion = tuple(torsion)
            if torsion not in self.index_dict:
                self.index_dict[torsion] = i

        improper_index_keys = [tuple(sorted(key)) for key in self.index_dict.keys()]
        # Now, reset all periodic torsion terms back to their initial values
        for pos, key in enumerate(self.torsion_store):
            try:
                self.tor_types[pos] = [
                    [key],
                    [float(self.torsion_store[key][i][1]) for i in range(4)],
                    [self.index_dict[key]],
                ]
            except KeyError:
                try:
                    self.tor_types[pos] = [
                        [tuple(reversed(key))],
                        [float(self.torsion_store[key][i][1]) for i in range(4)],
                        [self.index_dict[tuple(reversed(key))]],
                    ]
                except KeyError:
                    # after trying to match the forward and backwards strings must be improper
                    # now we have to work out the order it was stored in the system
                    improper_key = list(self.index_dict.keys())[
                        improper_index_keys.index(tuple(sorted(key)))
                    ]
                    self.tor_types[pos] = [
                        [improper_key],
                        [float(self.torsion_store[key][i][1]) for i in range(4)],
                        [self.index_dict[improper_key]],
                    ]

        self.update_torsions()

        # Reset the dihedral values
        self.tor_types = OrderedDict()

    def update_tor_vec(self, x):
        """Update the tor_types dict with the parameter vector."""
        # Round to 6 dp as this is the precision that will be in the xml files.
        x = np.round(x, decimals=6)

        # Update the param vector for the right torsions by slicing the vector every 4 places
        for key, val in self.tor_types.items():
            val[1] = x[key * 4 : key * 4 + 4]

    def objective(self, x):
        """Return the output of the objective function."""

        # Update the parameter vector into tor_types
        self.update_tor_vec(x)

        # Update the torsions in the OpenMM system
        self.update_torsions()

        # Get the mm corresponding energy
        self.mm_energy = deepcopy(self.mm_energies())

        # Make sure the energies match
        assert len(self.qm_energy) == len(self.mm_energy)

        # Calculate the objective
        # Make the mm energy relative to mm predicted energy of the qm optimised structure,
        # or lowest energy structure of scan
        if self.molecule.relative_to_global:
            mm_energy = self.mm_energy - self.open_mm.get_energy(self.opt_coords)
        else:
            mm_energy = self.mm_energy - self.mm_energy.min()

        error = (mm_energy - self.qm_energy) ** 2

        # if using a weighting, add that here
        if self.t_weight != "infinity":
            error *= np.exp(-self.qm_energy / (self.k_b * self.t_weight))

        # Find the total error
        total_error = np.sqrt(sum(error) / len(self.scan_coords))

        # Calculate the penalties
        # 1 the movement away from the starting values
        movement_penalty = self.l_pen * sum((x - self.starting_params) ** 2)

        # 2 the penalty incurred by going past the bounds
        bounds_penalty = sum(1 for vn in x if abs(vn) >= self.abs_bounds)

        total_error += movement_penalty + bounds_penalty
        return total_error

    def steep_objective(self, x):
        """Return the output of the objective function when using the steep refinement method."""

        # Update the parameter vector into tor_types
        self.update_tor_vec(x)

        # Update the torsions
        self.update_torsions()

        # first drive the torsion using geometric
        self.scan_coords = self.drive_mm("geometric")

        # Get the mm corresponding energy
        self.mm_energy = self.mm_energies()

        # Make sure the energies match
        assert len(self.qm_energy) == len(self.mm_energy)

        # calculate the objective

        # Adjust the mm energy to make it relative to the lowest in the scan
        self.mm_energy -= self.mm_energy.min()
        error = (self.mm_energy - self.qm_energy) ** 2

        # if using a weighting, add that here
        if self.t_weight != "infinity":
            error *= np.exp(-self.qm_energy / (self.k_b * self.t_weight))

        # Find the total error
        total_error = np.sqrt(sum(error) / len(self.scan_coords))

        # Calculate the penalty
        pen = self.l_pen * sum((x - self.starting_params) ** 2)

        total_error += pen
        return total_error

    def single_point_matching(self, fitting_error, opt_parameters):
        """A function the call the single point matching method of parameter refinement.

        method (fit only new generation)
        -------------------
        1) take parameters from the initial scipy fitting.
        2) Do a MM torsion scan with the parameters and get the rmsd error and energy error between this new surface
        and the qm optimised surface
        3) Now fit to the qm surface again using a small restrain penalty
        """

        converged = False

        # Set the optimisation method if we have a hybrd method we need to try and take the last option
        self.method = self.methods.get(self.molecule.opt_method.split("_")[-1], None)
        print(f"The optimisation method is {self.method}")

        # put in the objective dict
        objective = {
            "fitting_error": [],
            "energy_error": [],
            "rmsd": [],
            "total": [],
            "parameters": [],
        }

        iteration = 1
        # start the main optimizer loop by calculating new single point energies
        while not converged:
            # move into the first iteration folder
            make_and_change_into(f"Iteration_{iteration}")

            # step 2 MM torsion scan
            # with wavefront propagation, returns the new set of coords these become the new scan coords
            self.scan_coords = self.drive_mm("torsiondrive")

            # also save these coords to the coords store
            self.coords_store = deepcopy(self.coords_store + self.scan_coords)

            # step 3 calculate the rmsd for these structures compared to QM ones
            rmsd_vector = self.scan_rmsd(self.scan_coords)

            # Calculate how well the new relative surface represents the QM one
            energy_error = self.objective(opt_parameters)

            # this now acts as the intial energy for the next fit
            self.initial_energy = deepcopy(self.mm_energy)

            # add the results to the dictionary
            objective["fitting_error"].append(fitting_error)
            objective["energy_error"].append(energy_error)
            objective["rmsd"].append(sum(rmsd_vector) / len(rmsd_vector))
            objective["total"].append(
                energy_error + sum(rmsd_vector) / len(rmsd_vector)
            )
            objective["parameters"].append(opt_parameters)

            # Print the results of the iteration
            self.optimiser_log.write("After refinement the errors are:\n")
            for error, value in objective.items():
                self.optimiser_log.write(f"{error}: {value}\n")
            self.optimiser_log.flush()

            # Check convergence
            if objective["total"][-1] <= 0.25:
                print(f"Fitting converged after {iteration} iterations exiting...")
                # This takes us out of the refinement loop and stops any parameter changes
                break

            # Has the error converged?
            if iteration < 7:

                # Don't move too far away from the last set of optimised parameters if they got a good fit
                self.starting_params = opt_parameters
                # turn on the penalty if the error is getting close to the threshold
                if energy_error <= 1.5:
                    self.l_pen = 0.15
                else:
                    print("Turning off penalty due to large errors.")
                    self.l_pen = 0

                # optimise using the scipy method for the new structures with a penalty to remain close to the old
                fitting_error, opt_parameters = self.scipy_optimiser()

                # update the parameters in the fitting vector and the molecule for the MM scans
                self.update_tor_vec(opt_parameters)
                self.update_mol()

                # use the parameters to get the current energies
                self.mm_energy = deepcopy(self.mm_energies())

                self.optimiser_log.write(
                    f"Results for fitting iteration: {iteration}\n"
                )
                self.optimiser_log.flush()
                # plot the fitting graph this iteration
                self.plot_results(name=f"SP_iter_{iteration}")

                # move out of the folder
                os.chdir("../")

                # add 1 to the iteration
                iteration += 1

            else:
                # use the parameters to get the current energies
                self.mm_energy = deepcopy(self.mm_energies())
                # print the final iteration energy prediction
                self.plot_results(name=f"SP_iter_{iteration}")
                os.chdir("../")
                break

        # find the minimum total error index in list
        min_error = min(objective["total"])
        min_index = objective["total"].index(min_error)

        # gather the parameters with the lowest error, not always the last parameter set
        final_parameters = deepcopy(objective["parameters"][min_index])
        # final_parameters = deepcopy(objective['parameters'][-1])
        final_error = objective["total"][min_index]
        # final_error = objective['total'][-1]
        self.optimiser_log.write(
            f"The lowest error:{final_error}\nThe corresponding parameters:{final_parameters}\n"
            f"were found on iteraion {min_index + 1}\n"
        )
        self.optimiser_log.flush()

        # now we want to see how well we have captured the initial QM energy surface
        # reset the scan coords to the initial values
        self.scan_coords = self.initial_coords

        # Also save the last mm surface generated by the parameters
        final_surface_energy = deepcopy(self.mm_energy)

        # get the energy surface for these final parameters at the qm geometry
        energy_error = self.objective(final_parameters)
        self.optimiser_log.write(
            f"The final error at the qm optimised geometries is {energy_error}\n"
        )

        # get the starting energies back to the initial values before fitting
        self.initial_energy = self.starting_energy
        # plot the results this is a graph of the starting QM surface and how well we can remake it
        self.optimiser_log.write("The final stage 2 fitting results:\n")
        self.optimiser_log.flush()

        self.plot_results(
            name="Stage2_Single_point_fit",
            extra_points={"Final parameters MM geometry": final_surface_energy},
        )

        # Plot the convergence of the energy rmsd and total errors
        self.plot_convergence(objective)

        # Plot the correlation between the single point energies over all structures sampled in the fitting
        # Using the initial and final parameters
        self.plot_correlation(final_parameters)

        return final_error, final_parameters

    def qm_normalise(self):
        """
        Normalize the qm energy to the reference energy which is either the lowest in the set or the global minimum
        :return: normalised qm vector
        """

        if self.molecule.relative_to_global:
            self.qm_energy -= self.molecule.qm_energy
            print("Using the optimised structure!")

        else:
            # Make relative to lowest energy
            self.qm_energy -= self.qm_energy.min()

        self.qm_energy *= constants.HA_TO_KCAL_P_MOL

    def torsion_test(self):
        """
        Take optimized xml file and test the agreement with QM by doing a torsion drive and checking the single
        point energies for each rotatable dihedral.
        """

        # Run the scanner
        for i, self.scan in enumerate(self.molecule.scan_order):
            # move into the scan folder that should have been made
            make_and_change_into(f"SCAN_{self.scan[0]}_{self.scan[1]}")

            # Move into testing folder
            try:
                rmtree("testing_torsion")
            except FileNotFoundError:
                pass

            make_and_change_into("testing_torsion")

            # Run torsiondrive
            # step 2 MM torsion scan
            # with wavefront propagation, returns the new set of coords these become the new scan coords
            self.scan_coords = self.drive_mm("torsiondrive")

            # step 4 calculate the single point energies
            self.qm_energy = self.single_point()

            # Normalise the qm energy again using the qm reference energy
            self.qm_normalise()

            # Calculate the mm energy
            self.reset_torsions()
            # Use the parameters to get the current energies
            self.mm_energy = deepcopy(self.starting_energy)

            # Graph the energy
            self.plot_results(name="testing_torsion", torsion_test=True)

            os.chdir("../../")

    def run(self):
        """
        Optimise the parameters for the chosen torsions in the molecule scan_order,
        also set up a work queue to do the single point calculations if they are needed.
        """

        # Set up the first fitting
        for self.scan in self.molecule.scan_order:
            # Get the MM coords from the QM torsion drive in OpenMM format
            self.molecule.coordinates = self.molecule.qm_scans[self.scan][1]
            self.scan_coords = self.molecule.openmm_coordinates()
            self._create_rdkit_molecules(self.scan_coords)
            # Set up the fitting folders
            scan_name = f"SCAN_{self.scan[0]}_{self.scan[1]}"
            try:
                os.mkdir(scan_name)
            except FileExistsError:
                # back up the old scan data
                os.rename(
                    scan_name,
                    f"{scan_name}_backup",
                )
                os.mkdir(scan_name)

            # Make and move through the folders  and create the log file
            os.chdir(scan_name)
            self.optimiser_log = open("Optimiser_log.txt", "w")
            self.optimiser_log.write(
                f'Starting dihedral optimisation at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.\n'
            )
            self.optimiser_log.write(
                f"Optimising dihedrals for central bond {self.scan}\n"
            )
            self.optimiser_log.flush()
            os.mkdir("First_fit")
            os.mkdir("Refinement")
            os.chdir("First_fit")

            # Set the target energies first
            self.target_energy = self.scans_dict[self.scan][0]

            # Adjust the QM energies
            # and store all QM raw energies
            self.energy_store_qm = deepcopy(self.target_energy)
            self.qm_energy = deepcopy(self.target_energy)
            # store the optimized qm energy and make all other energies relative to this one

            self.qm_normalise()

            # Keep the initial coords
            self.coords_store = deepcopy(self.scan_coords)
            self.initial_coords = deepcopy(self.scan_coords)

            # Reset all torsions to their initial values
            self.reset_torsions()

            # Get the torsions that will be fit and make the param vector
            self.get_torsion_params()

            # Now measure the starting objective function
            starting_error = deepcopy(self.objective(self.starting_params))
            # initial is a reference to the energy surface at the start of the fit
            self.initial_energy = deepcopy(self.mm_energy)
            # starting energy is the surface made by the original unfit parameters
            self.starting_energy = deepcopy(self.initial_energy)

            # Start the main optimiser loop and get the final error and parameters back
            self.optimiser_log.write(
                f"Starting initial optimisation\nThe starting error is: {starting_error}\n"
            )
            append_to_log(
                self.molecule.home,
                f"Initial error of {self.scan}, before dihedral optimisation: {starting_error}",
                and_print=True,
            )
            self.optimiser_log.flush()
            # Get the optimisation method if it is a hybrid take the first option
            self.method = self.methods.get(self.molecule.opt_method.split("_")[0], None)
            error, opt_parameters = self.scipy_optimiser()
            self.optimiser_log.write(f"fitted parameters {opt_parameters}\n")
            self.optimiser_log.write(f"Fitted error {error}\n")
            append_to_log(
                self.molecule.home,
                f"Fitted error for {self.scan}: {error}",
                and_print=True,
            )
            self.optimiser_log.flush()

            self.param_vector = opt_parameters

            # Push the new parameters back to the molecule parameter dictionary
            self.update_mol()

            self.optimiser_log.write("Optimisation finished\n")
            self.optimiser_log.flush()
            # Plot the results of the first fit
            self.plot_results(name="Stage1_scipy")

            # move to the refinement section
            os.chdir("../Refinement")

            if self.refinement == "SP":
                self.optimiser_log.write(
                    "Starting refinement method single point matching\n"
                )
                self.optimiser_log.flush()
                error, opt_parameters = self.single_point_matching(
                    error, opt_parameters
                )
                self.param_vector = opt_parameters

            # now push the parameters back to the molecule
            self.update_tor_vec(opt_parameters)
            self.update_mol()

            # now move back to the starting directory
            os.chdir(self.home)

            self.optimiser_log.write(
                f'Optimisation finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.\n'
            )
            self.optimiser_log.flush()
            self.optimiser_log.close()

    def initial_optimiser(self):
        """This is the initial optimiser which is ran on the qm optimised geometry and determines the best
        fitting parameters to be used in the other optimisations based on the current optimisation method.
        1) Fit with no temperature weighting using the optimiser and measure the error.
        2) Determine if the temperature should be changed when the fit is poor and there are large high energy
        peaks on the surface
        3) Fit with the new temperature store the settings of the lowest error."""

        running_options = {}
        for self.t_weight in ["infinity"]:
            error, opt_parameters = self.scipy_optimiser()
            running_options[self.t_weight] = (error, opt_parameters)
            if error <= 1.5:
                return error, opt_parameters
            else:
                print(
                    f"Current error {error} adjusting temperature wieghting to improve fitting"
                )

        # If we could not get the error below the threshold then return that with the lowest error
        self.t_weight, options = min(running_options.items(), key=lambda x: x[1][0])
        return options[0], options[1]

    def load_torsions(self):
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
            if self.molecule.PeriodicTorsionForce[key][-1] == "Improper":
                self.molecule.PeriodicTorsionForce[key] = [
                    ["1", "1", "0"],
                    ["2", "1", f"{constants.PI}"],
                    ["3", "1", "0"],
                    ["4", "1", f"{constants.PI}"],
                    "Improper",
                ]
            else:
                self.molecule.PeriodicTorsionForce[key] = [
                    ["1", "1", "0"],
                    ["2", "1", f"{constants.PI}"],
                    ["3", "1", "0"],
                    ["4", "1", f"{constants.PI}"],
                ]

        # Write out the new xml file which is read into the OpenMM system
        self.molecule.write_parameters()

        # Put the torsions back into the molecule
        self.molecule.PeriodicTorsionForce = deepcopy(self.torsion_store)

    def get_torsion_params(self):
        """
        Get the torsions and their parameters that will be scanned;
        work out how many different torsion types are needed;
        make a vector corresponding to this size.
        """

        # Get a list of which dihedrals parameters are to be varied through the scanned bond
        try:
            to_fit = [tuple(tor) for tor in list(self.molecule.dihedrals[self.scan])]
        except KeyError:
            # here we must have an improper torsion for now only take it to fit
            # TODO extend dihedral types to cover symmetry equivalent impropers
            to_fit = [self.scan]
        # Now expand to include all symmetry equivalent dihedrals
        torsions_and_types = {}

        for key, dihedral_class in self.molecule.dihedral_types.items():
            for dihedral in dihedral_class:
                if dihedral in to_fit or tuple(reversed(dihedral)) in to_fit:
                    torsions_and_types[key] = dihedral_class
                    # once the whole class is added move to the next
                    break

        # Check which ones have the same parameters and how many torsion vectors we need
        self.tor_types = OrderedDict()

        # now populate the tor_types dict eg {0: [[(torsion key)], [starting param vector]
        # [openMM system index of first torsion], [symmetry type]]
        i = 0
        for symmetry_key, torsions in torsions_and_types.items():
            torsion = torsions[0]
            try:
                master_vector = [
                    float(self.torsion_store[torsion][i][1]) for i in range(4)
                ]
            except KeyError:
                torsion = torsion[::-1]
                master_vector = [
                    float(self.torsion_store[torsion][i][1]) for i in range(4)
                ]

            self.tor_types[i] = [torsions, master_vector, [], symmetry_key]
            # Now add in all of the openmm system parameter index
            for torsion in torsions:
                try:
                    self.tor_types[i][2].append(self.index_dict[torsion])
                except KeyError:
                    self.tor_types[i][2].append(
                        self.index_dict[tuple(reversed(torsion))]
                    )

            i += 1

        self.param_vector = np.array(
            [0 for _ in range(4) for _ in range(len(self.tor_types))]
        )

        # now take the master vectors and make the starting parameter list
        # Store the original parameter vectors to use regularisation
        self.starting_params = np.array(
            [list(k)[1][i] for k in self.tor_types.values() for i in range(4)]
        )
        # Work out what the torsion parameter limit based on the old parameters
        # should be similar if we have very large barrier heights
        if self.abs_bounds <= self.starting_params.max():
            self.abs_bounds = round(self.starting_params.max() + 2)
        self.optimiser_log.write(f"Starting parameters {self.starting_params}\n")
        self.optimiser_log.flush()

    def _create_rdkit_molecules(self, coordinates):
        """
        Create a list of rdkit molecules corresponding to the coordinates of the qm torsion scan,
        these are used to compute the rmsd.
        :param coordinates: A list of numpy arrays used to generate the conformers
        :return: a list of rdkit molecules each corresponding to a point on the torsionscan
        """
        from qubekit.molecules.utils import RDKit

        self.rmsd_atoms = []
        for coord in coordinates:
            rdkit_mol = deepcopy(self.molecule.to_rdkit())
            rdkit_mol.RemoveAllConformers()
            rdkit_mol = RDKit.add_conformer(
                rdkit_mol, np.array(coord) * constants.NM_TO_ANGS
            )
            self.rmsd_atoms.append(rdkit_mol)

    def scan_rmsd(self, coordinates):
        """
        Calculate the rmsd between the stored qm positions and the new set of coordinates
        :param coordinates:  A list of numpy coordinate arrays
        :return: a list containing the rmsd values for each pair of coordinates
        """
        from qubekit.molecules.utils import RDKit

        # Make sure the number of coordinates we pass is the same as the number of reference positions that we have
        if len(coordinates) != len(self.rmsd_atoms):
            print(
                f"len(coordinates): {len(coordinates)};  len(self.rmsd_atoms): {len(self.rmsd_atoms)}"
            )
        assert len(coordinates) == len(self.rmsd_atoms)

        rmsd = []
        for coord, molecule in zip(coordinates, self.rmsd_atoms):
            molecule = RDKit.add_conformer(
                molecule, np.array(coord) * constants.NM_TO_ANGS
            )
            rmsd.append(
                RDKit.get_conformer_rmsd(molecule, 0, molecule.GetNumConformers() - 1)
            )

        return rmsd

    def finite_difference(self, x):
        """Compute the gradient of changing the parameter vector using central difference scheme."""

        gradient = []
        for i in range(len(x)):
            x[i] += self.step_size / 2
            plus = self.objective(x)
            x[i] -= self.step_size
            minus = self.objective(x)
            diff = (plus - minus) / self.step_size
            gradient.append(diff)

        return np.array(gradient)

    def scipy_optimiser(self):
        """The main torsion parameter optimiser that controls the optimisation method used."""

        print(f"Running SciPy {self.method} optimiser ... ")
        # TODO all methods should use bounds which can control the amount of vn we fit

        if self.method == "Nelder-Mead":
            res = minimize(
                self.objective,
                self.param_vector,
                method=self.method,
                options={
                    "xtol": self.x_tol,
                    "ftol": self.error_tol,
                    "disp": True,
                    "maxiter": 10000,
                },
            )

        elif self.method == "BFGS":
            res = minimize(
                self.objective,
                self.param_vector,
                method=self.method,
                jac=self.finite_difference,
                options={"disp": True},
            )

        elif self.method == "Genetic":
            # We must create some bounds for this method based on the vn_limits
            bounds = [
                (-abs(self.abs_bounds), abs(self.abs_bounds))
                for _ in range(len(self.param_vector))
            ]
            res = differential_evolution(self.objective, bounds=bounds)

        else:
            raise NotImplementedError(
                "This optimisation method is not implemented; options are: Nelder-Mead, BFGS, Genetic."
            )

        print("SciPy optimisation complete")

        # Update the tor types dict using the optimised vector
        self.update_tor_vec(res.x)

        # return the final fitting error and final param vector after the optimisation
        return res.fun, res.x

    def update_torsions(self):
        """Update the torsions being fitted."""

        forces = {
            self.open_mm.simulation.system.getForce(
                index
            ).__class__.__name__: self.open_mm.simulation.system.getForce(index)
            for index in range(self.open_mm.simulation.system.getNumForces())
        }
        torsion_force = forces["PeriodicTorsionForce"]

        for val in self.tor_types.values():
            for j, dihedral in enumerate(val[0]):
                for v_n in range(4):
                    *dihedral, _, _, _ = torsion_force.getTorsionParameters(
                        v_n + val[2][j]
                    )
                    torsion_force.setTorsionParameters(
                        index=v_n + val[2][j],
                        periodicity=v_n + 1,
                        phase=self.phases[v_n],
                        k=val[1][v_n],
                        particle1=dihedral[0],
                        particle2=dihedral[1],
                        particle3=dihedral[2],
                        particle4=dihedral[3],
                    )
        torsion_force.updateParametersInContext(self.open_mm.simulation.context)

    def plot_results(self, name="Plot", torsion_test=False, extra_points=None):
        """
        Plot the results of the scan.
        :param name: Name of the pdf made
        :param torsion_test: changes the style of the plot for testing
        :param extra_points: a dictionary of label's and extra points to plot
        :return:
        """

        # Make sure we have the same number of energy terms in the QM and MM lists
        assert len(self.qm_energy) == len(self.mm_energy)

        # Adjust the MM energies
        if self.molecule.relative_to_global:
            if torsion_test:
                initial_energy = self.mm_energy - self.open_mm.get_energy(
                    self.opt_coords
                )
                plot_mm_energy = initial_energy
            else:
                plot_mm_energy = self.mm_energy - self.open_mm.get_energy(
                    self.opt_coords
                )
                initial_energy = self.initial_energy - self.open_mm.get_energy(
                    self.opt_coords
                )

            if extra_points is not None:
                for key, val in extra_points.items():
                    extra_points[key] = val - self.open_mm.get_energy(self.opt_coords)

        else:
            if torsion_test:
                initial_energy = self.mm_energy - self.mm_energy.min()
                plot_mm_energy = initial_energy
            else:
                plot_mm_energy = self.mm_energy - self.mm_energy.min()
                initial_energy = self.initial_energy - self.initial_energy.min()

            if extra_points is not None:
                for key, val in extra_points.items():
                    extra_points[key] = val - val.min()

        # Construct the angle array
        angles = list(
            range(
                self.molecule.dih_starts[self.scan],
                self.molecule.dih_ends[self.scan] + self.molecule.increments[self.scan],
                self.molecule.increments[self.scan],
            )
        )

        # Make sure we have the same angles as data points
        assert len(angles) == len(self.qm_energy)
        # Print a table of the results
        if torsion_test:
            self.optimiser_log.write(
                f"Angle    QM(relative)        MM_initial(relative)\n"
            )
            self.optimiser_log.flush()
            for data in zip(angles, self.qm_energy, initial_energy):
                self.optimiser_log.write(
                    f"{data[0]:4}  {data[1]:15.10f}     {data[2]:15.10f}\n"
                )
                self.optimiser_log.flush()
        else:
            self.optimiser_log.write(
                f"Angle    QM(relative)        MM(relative)    MM_initial(relative)\n"
            )
            self.optimiser_log.flush()
            for data in zip(angles, self.qm_energy, plot_mm_energy, initial_energy):
                self.optimiser_log.write(
                    f"{data[0]:4}  {data[1]:15.10f}     {data[2]:15.10f}    {data[3]:15.10f}\n"
                )
                self.optimiser_log.flush()

        plt.xlabel(r"Dihedral angle$^{\circ}$")

        # Plot the qm and mm data
        plt.plot(angles, self.qm_energy, "o", label="QM data")
        if torsion_test:
            plt.plot(angles, initial_energy, label="Current parameters")
        else:
            plt.plot(
                angles, initial_energy, label="Starting parameters", linestyle="--"
            )
            plt.plot(angles, plot_mm_energy, label="Final parameters")
            if extra_points is not None:
                for title, values in extra_points.items():
                    plt.plot(angles, values, label=title)

        # Label the graph and save the pdf
        mol_di = self.molecule.dihedrals[self.scan[1:3]][0]
        plt.title(
            f"Relative energy surface for dihedral {mol_di[0]}-{mol_di[1]}-{mol_di[2]}-{mol_di[3]}"
        )
        plt.ylabel("Relative energy (kcal/mol)")
        plt.legend(loc=1)
        plt.savefig(f"{name}.pdf")
        plt.clf()

    def plot_correlation(self, optimised_parameters):
        """Plot the single point energy correlation between all points in the fitting"""

        # First get every qm energy and coordinate from the stores and measure all energies relative to
        # the qm optimised structure
        # rel_to_global = deepcopy(self.molecule.relative_to_global)
        # self.molecule.relative_to_global = True
        # self.qm_energy = deepcopy(self.energy_store_qm)
        # self.qm_normalise()

        start_phi = self.molecule.dih_starts[self.scan]
        end_phi = self.molecule.dih_ends[self.scan]
        n_steps = int((end_phi - start_phi) / (self.molecule.increments[self.scan])) + 1

        self.scan_coords = deepcopy(self.coords_store[:n_steps])

        # Calculate the mm energies of all of the structures with the new torsion parameters
        self.objective(optimised_parameters)

        mm_energy_new = self.mm_energy - self.mm_energy.min()
        slope_new, intercept_new, r_value_new, _, _ = linregress(
            self.qm_energy, mm_energy_new
        )

        # Calculate the error using the old parameters
        self.objective(self.starting_params)
        mm_energy_old = self.mm_energy - self.mm_energy.min()
        slope_old, intercept_old, r_value_old, _, _ = linregress(
            self.qm_energy, mm_energy_old
        )

        # Make sure we have the same number of energy terms in the QM and MM lists
        assert len(self.qm_energy) == len(mm_energy_new) == len(mm_energy_old)

        # now we are just plotting them against each other they are already in the right order
        plt.scatter(
            self.qm_energy,
            mm_energy_old,
            label=rf"starting parameters $r^2$={r_value_old ** 2:.4f}",
            s=10,
        )
        plt.scatter(
            self.qm_energy,
            mm_energy_new,
            label=rf"Final parameters $r^2$={r_value_new ** 2:.4f}",
            s=10,
        )
        x = np.linspace(self.qm_energy.min(), self.qm_energy.max(), 100)
        mm_min = min(mm_energy_new.min(), mm_energy_old.min())
        mm_max = max(mm_energy_new.max(), mm_energy_old.max())
        y = np.linspace(mm_min, mm_max, 100)
        # Line x = y
        plt.plot(x, y, ls="--", c=".3", zorder=2)

        plt.xlabel("Relative energy (kcal/mol) QM energy")
        plt.ylabel("Relative energy (kcal/mol) MM energy")
        plt.legend(loc=1)
        plt.savefig(f"correlation_plot.pdf")
        plt.clf()

        # reset the relative to global setting
        # self.molecule.relative_to_global = rel_to_global

    def plot_convergence(self, objective):
        """
        Plot the convergence of the errors through the iterative fitting methods
        :param objective: A dictionary containing all of the error measurements
        """

        iterations = [x for x in range(len(objective["total"]))]
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Fitting iteration")
        ax1.set_ylabel("Error (kcal / mol)", color=color)
        ax1.plot(
            iterations,
            objective["energy_error"],
            label="Energy error",
            color=color,
            linestyle="--",
        )
        ax1.plot(iterations, objective["total"], label="Total error", color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        color = "tab:blue"
        ax2 = ax1.twinx()
        ax2.plot(iterations, objective["rmsd"], label="RMSD error", color=color)
        ax2.set_ylabel("RMSD", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        # get the plotted lines and labels from MatPlotLib to combine the legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=1)

        fig.tight_layout()
        plt.savefig(f"convergence.pdf")
        plt.clf()

    def make_constraints(self):
        """Write a constraint file used by geometric during optimizations."""

        with open("qube_constraints.txt", "w+") as constraint:
            mol_di = self.scan
            start_phi = self.molecule.dih_starts[mol_di]
            end_phi = self.molecule.dih_ends[mol_di]
            n_steps = (
                int((end_phi - start_phi) / (self.molecule.increments[mol_di])) + 1
            )
            constraint.write(
                f"$scan\ndihedral {mol_di[0]} {mol_di[1]} {mol_di[2]} {mol_di[3]} {start_phi} {end_phi} {n_steps}\n"
            )

            if self.molecule.constraints_file:
                with open(self.molecule.constraints_file) as cons_file:
                    for line in cons_file:
                        constraint.write(line)

    def write_dihedrals(self):
        """Write out the torsion drive dihedral file for the current self.scan."""

        with open("dihedrals.txt", "w+") as out:
            out.write(
                "# dihedral definition by atom indices starting from 0\n#zero_based_numbering"
                "\n# i     j     k     l\n"
            )

            mol_di = self.scan
            start_phi = self.molecule.dih_starts[mol_di]
            end_phi = self.molecule.dih_ends[mol_di]
            out.write(
                f"  {mol_di[0]}     {mol_di[1]}     {mol_di[2]}     {mol_di[3]}     {start_phi}  {end_phi}\n"
            )

    def drive_mm(self, engine):
        """Drive the torsion again using MM to get new structures."""

        # Write an xml file with the new parameters

        # Move into a temporary folder torsion drive gives an error if we use temp directory module
        temp = f"{engine}_scan"
        try:
            rmtree(temp)
        except FileNotFoundError:
            pass

        make_and_change_into(temp)

        # Write out a pdb file of the qm optimised geometry
        self.molecule.to_file(file_name="openmm.pdb")
        # Also need an xml file for the molecule to use in geometric
        self.molecule.write_parameters(name="openmm")
        # openmm.pdb and input.xml are the expected names for geometric
        with open("log.txt", "a+") as log:
            if engine == "torsiondrive":
                if self.molecule.constraints_file is not None:
                    os.system("mv ../constraints.txt .")
                self.write_dihedrals()

                step_size = self.molecule.increments[self.scan]

                sp.run(
                    f"torsiondrive-launch -e openmm openmm.pdb dihedrals.txt -v -g {step_size}"
                    f' {self.molecule.constraints_file if self.molecule.constraints_file is not None else ""}',
                    shell=True,
                    stderr=log,
                    stdout=log,
                    check=True,
                )

                self.molecule.read_tdrive(self.scan)
                positions = self.molecule.qm_scans[self.scan][1]

            elif engine == "geometric":
                if self.molecule.constraints_file is not None:
                    os.system("mv ../constraints.txt .")
                else:
                    self.make_constraints()

                sp.run(
                    "geometric-optimize --epsilon 0.0 --maxiter 500 --qccnv true --pdb openmm.pdb "
                    "--engine openmm state.xml qube_constraints.txt",
                    shell=True,
                    stdout=log,
                    stderr=log,
                    check=True,
                )

            else:
                raise NotImplementedError(
                    "Invalid torsion engine. Please use torsiondrive or geometric"
                )

        # move back to the master folder
        os.chdir("../")

        return positions

    # def single_point(self):
    #     """Take set of coordinates of a molecule and do a single point calculation; returns an array of the energies."""
    #
    #     # reset the temp entry in the molecule
    #     self.molecule.coords["temp"] = self.molecule.coords["input"]
    #     # for each coordinate in the system we need to write a qm input file and get the single point energy
    #     try:
    #         rmtree("Single_points")
    #     except FileNotFoundError:
    #         pass
    #
    #     make_and_change_into("Single_points")
    #
    #     sp_energy = []
    #
    #     for i, x in enumerate(self.scan_coords):
    #         make_and_change_into(f"SP_{i}")
    #         print(
    #             f"Doing single point calculations on new structures ... {i + 1}/{len(self.scan_coords)}"
    #         )
    #         # Change the positions of the molecule in the molecule array
    #         for y, coord in enumerate(x):
    #             for z, pos in enumerate(coord):
    #                 # Convert from nanometers in openmm to Angs in QM and store in the temp position in the molecule
    #                 self.qm_engine.molecule.coords["temp"][y][z] = (
    #                     pos * constants.NM_TO_ANGS
    #                 )
    #
    #         # Write the new coordinate file and run the calculation
    #         result = self.qm_engine.generate_input(input_type="temp", energy=True)
    #         if result["success"] and self.qm_engine.__class__.__name__ == "Gaussian":
    #             coords, energy = self.qm_engine.optimised_structure()
    #             sp_energy.append(energy)
    #
    #         elif result["success"] and self.qm_engine.__class__.__name__ == "PSI4":
    #             # Extract the energy and save to the array
    #             sp_energy.append(PSI4.get_energy())
    #
    #         # Move back to the base directory
    #         os.chdir("../")
    #
    #     # move out to the main folder
    #     os.chdir("../")
    #
    #     return np.array(sp_energy)

    def update_mol(self):
        """When the optimisation is complete, update the PeriodicTorsionForce parameters in the molecule."""

        for val in self.tor_types.values():
            for dihedral in val[0]:
                for vn in range(4):
                    try:
                        self.molecule.PeriodicTorsionForce[dihedral][vn][1] = str(
                            val[1][vn]
                        )
                    except KeyError:
                        self.molecule.PeriodicTorsionForce[tuple(reversed(dihedral))][
                            vn
                        ][1] = str(val[1][vn])
