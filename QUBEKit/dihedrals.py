#!/usr/bin/env python3

import math
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

from QUBEKit.engines import PSI4, Gaussian, OpenMM, RDKit
from QUBEKit.ligand import Ligand
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import TorsionDriveData
from QUBEKit.utils.exceptions import TorsionDriveFailed
from QUBEKit.utils.file_handling import make_and_change_into

matplotlib.use("Agg")  # Fix for clusters?


class TorsionScan:
    """
    This class will take a QUBEKit molecule object and perform a torsiondrive QM energy scan
    for each selected rotatable dihedral.

    inputs
    ---------------
    molecule                A QUBEKit Ligand instance
    constraints_made        The name of the constraints file that should be used during the torsiondrive (pis4 only)

    attributes
    ---------------
    qm_engine               An instance of the QM engine used for any calculations
    native_opt              Chosen dynamically whether to use geometric or not (geometric is need to use constraints)
    input_file               The name of the template file for tdrive, name depends on the qm_engine used
    home                    The starting location of the job, helpful when scanning multiple angles.
    """

    def __init__(self, molecule: Ligand, constraints_made=None):

        self.molecule = molecule
        self.molecule.convergence = "GAU"
        self.constraints_made = constraints_made

        self.qm_engine = {"psi4": PSI4, "g09": Gaussian, "g16": Gaussian}.get(
            molecule.bonds_engine
        )(molecule)
        self.native_opt = True

        # Ensure geometric can only be used with psi4 so far
        if molecule.geometric and molecule.bonds_engine == "psi4":
            self.native_opt = False

        self.input_file = None

        self.home = os.getcwd()

    def is_short_cc_bond(self, bond):
        atom_0, atom_1 = [self.molecule.rdkit_mol.GetAtomWithIdx(atom) for atom in bond]

        if atom_0.GetSymbol().upper() == "C" and atom_1.GetSymbol().upper() == "C":
            if atom_0.GetDegree() == 3 and self.molecule.measure_bonds[bond] < 1.42:
                return True
        return False

    def find_scan_order(self):
        """
        Function takes the molecule and displays the rotatable central bonds,
        the user then enters the numbers of the torsions to be scanned (in the order they'll be scanned in).
        The molecule can also be supplied with a scan order already, if coming from csv.
        Else the user can supply a torsiondrive style QUBE_torsions.txt file we can extract the parameters from.
        """

        if self.molecule.scan_order:
            return

        # Get the rotatable dihedrals from the molecule
        self.molecule.scan_order = []
        if self.molecule.rotatable_bonds is None:
            return

        rotatable = set()
        non_rotatable = set()

        for dihedral_class in self.molecule.dihedral_types.values():
            dihedral = dihedral_class[0]
            bond = dihedral[1:3]
            if bond in self.molecule.rotatable_bonds:
                rotatable.add(bond)
            else:
                non_rotatable.add(dihedral)

        for bond in rotatable:
            dihedral = self.molecule.dihedrals[bond][0]
            if self.is_short_cc_bond(bond):
                non_rotatable.add(dihedral)
                continue

            self.molecule.dih_starts[dihedral] = -165
            self.molecule.dih_ends[dihedral] = 180
            rev_dihedral = tuple(reversed(dihedral))
            self.molecule.dih_starts[rev_dihedral] = -165
            self.molecule.dih_ends[rev_dihedral] = 180
            self.molecule.scan_order.append(dihedral)
            self.molecule.increments[dihedral] = 15
            self.molecule.increments[rev_dihedral] = 15

        if hasattr(self.molecule, "skip_nonrotatable"):
            if self.molecule.skip_nonrotatable == "no":
                for dihedral in non_rotatable:
                    bond = dihedral[1:3]
                    restricted, half_restricted = False, False
                    atom_0 = self.molecule.rdkit_mol.GetAtomWithIdx(bond[0])
                    if (
                        atom_0.IsInRingSize(3)
                        or atom_0.IsInRingSize(4)
                        or atom_0.IsInRingSize(5)
                    ):
                        restricted = True
                    if not restricted:
                        if self.is_short_cc_bond(bond):
                            restricted = True

                    current_val = self.molecule.measure_dihedrals()[dihedral]
                    if restricted:
                        atom_0 = self.molecule.rdkit_mol.GetAtomWithIdx(dihedral[0])
                        atom_3 = self.molecule.rdkit_mol.GetAtomWithIdx(dihedral[3])
                        half_restricted = not (atom_0.IsInRing() and atom_3.IsInRing())

                    if restricted:
                        # first or last atom are not part of a ring
                        if half_restricted:
                            self.molecule.dih_starts[dihedral] = int(current_val - 45.0)
                            self.molecule.dih_ends[dihedral] = int(current_val + 45.0)
                        # first and last part of a ring -> be careful
                        else:
                            self.molecule.dih_starts[dihedral] = int(current_val - 15.0)
                            self.molecule.dih_ends[dihedral] = int(current_val + 15.0)

                    else:
                        if abs(current_val) <= 90:
                            self.molecule.dih_starts[dihedral] = -100
                            self.molecule.dih_ends[dihedral] = 100
                        else:
                            self.molecule.dih_starts[dihedral] = 80
                            self.molecule.dih_ends[dihedral] = 280

                    rev_dihedral = tuple(reversed(dihedral))
                    self.molecule.dih_starts[rev_dihedral] = self.molecule.dih_starts[
                        dihedral
                    ]
                    self.molecule.dih_ends[rev_dihedral] = self.molecule.dih_ends[
                        dihedral
                    ]
                    self.molecule.scan_order.append(dihedral)
                    self.molecule.increments[dihedral] = 5
                    self.molecule.increments[rev_dihedral] = 5

        self.molecule.scan_order.reverse()
        print(f"scan order: {self.molecule.scan_order}")

    def tdrive_scan_input(self, scan):
        """Function takes the rotatable dihedrals requested and writes a scan input file for torsiondrive."""

        scan_di = scan
        if self.molecule.improper_torsions is None and len(scan) == 2:
            scan_di = self.molecule.dihedrals[scan][0]

        # Write the dihedrals.txt file for tdrive
        with open("dihedrals.txt", "w+") as out:

            out.write(
                "# dihedral definition by atom indices starting from 0\n# zero_based_numbering\n"
                "# i     j     k     l     "
            )
            out.write("(range_low)     (range_high)\n")
            out.write(
                f"  {scan_di[0]}     {scan_di[1]}     {scan_di[2]}     {scan_di[3]}     "
            )
            out.write(
                f"{self.molecule.dih_starts[scan_di]}     {self.molecule.dih_ends[scan_di]}\n"
            )

        # Then write the template input file for tdrive in g09 or psi4 format
        if self.native_opt:
            self.qm_engine.generate_input(optimise=True, execute=False)
            if self.qm_engine.__class__.__name__.lower() == "psi4":
                self.input_file = "input.dat"
            else:
                self.input_file = f"gj_{self.molecule.name}.com"

        else:
            self.qm_engine.geo_gradient(execute=False, threads=True)
            self.input_file = (
                f"{self.molecule.name}.{self.qm_engine.__class__.__name__.lower()}in"
            )

    def start_torsiondrive(self, scan):
        """Start a torsiondrive either using psi4 or native gaussian09"""

        # TODO maybe we should be using the json api to have more control over the engine settings
        #  we could also run multiple grid points at the same time, progress reporting would also be
        #  better

        # First set up the required files
        self.tdrive_scan_input(scan)

        # Now we need to run torsiondrive through the CLI
        with open("tdrive.log", "w") as log:
            # When we start the run write the options used here to be used during restarts
            log.write(
                f"Theory used: {self.molecule.theory}   Basis used: {self.molecule.basis}\n"
            )
            log.flush()

            if self.molecule.bonds_engine == "g09":
                tdrive_engine = "gaussian09"
            elif self.molecule.bonds_engine == "g16":
                tdrive_engine = "gaussian16"
            else:
                tdrive_engine = self.molecule.bonds_engine

            span = self.molecule.dih_ends[scan] - self.molecule.dih_starts[scan]
            step_size = 10
            if span / step_size < 15:
                step_size = span / 35
                step_size = 5 * math.ceil(step_size / 5)

            self.molecule.increments[scan] = step_size

            cmd = (
                f"torsiondrive-launch -e {tdrive_engine} {self.input_file} dihedrals.txt -v -g {step_size}"
                f'{" --native_opt" if self.native_opt else ""}'
            )

            if not os.path.exists("qdata.txt"):
                sp.run(cmd, shell=True, stdout=log, check=True, stderr=log, bufsize=0)

        if self.molecule.bonds_engine in ["g09", "g16"]:
            Gaussian.cleanup()

        # Gather the results
        try:
            result = TorsionDriveData.from_qdata(dihedral=scan)
            self.molecule.add_qm_scan(scan_data=result)
            # self.molecule.read_tdrive(scan)
        except FileNotFoundError as exc:
            if not self.molecule.tdrive_parallel:
                raise TorsionDriveFailed(
                    "Torsiondrive output qdata.txt missing; job did not execute or finish properly"
                ) from exc

    def collect_scans(self):
        """
        Collect the results of a torsiondrive scan that has not been done using QUBEKit.
        :return: The energies and coordinates into the molecule
        """

        for scan in self.molecule.scan_order:
            name = self._make_folder_name(scan)
            result = TorsionDriveData.from_qdata(
                dihedral=scan,
                qdata_file=os.path.join(
                    self.home, name, "QM_torsiondrive", "qdata.txt"
                ),
            )
            self.molecule.add_qm_scan(scan_data=result)

    def _make_folder_name(self, scan):
        return f"SCAN_{scan[0]}_{scan[1]}_{scan[2]}_{scan[3]}"

    def check_run_history(self, scan):
        """
        Check the contents of a scan folder to see if we should continue the run;
        if the settings (theory and basis) are the same then continue
        :return: If we should continue the run
        """

        name = self._make_folder_name(scan)
        # Try and open the tdrive.log file to check the old running options
        try:
            file_path = os.path.join(os.getcwd(), name, "QM_torsiondrive", "tdrive.log")
            with open(file_path) as t_log:
                header = t_log.readline().split("Basis used:")

            # This is needed for the case of a split theory e.g. EmpiricalDispersion=GD3BJ B3LYP
            basis = header[1].strip()
            theory = header[0].split("Theory used:")[1].strip()
            return self.molecule.theory == theory and self.molecule.basis == basis

        except FileNotFoundError:
            return False

    def scan(self):
        """Makes a folder and writes a new a dihedral input file for each scan and runs the scan."""

        # TODO QCArchive/Fractal search; don't do a calc that has been done!

        # TODO
        #   if the molecule has multiple scans to do they should all start at the same time as this is slow
        #   We must also make sure that we don't exceed the core limit when we do this!
        #   e.g. user gives 6 cores for QM and we run two drives that takes 12 cores!

        for scan in self.molecule.scan_order:
            name = self._make_folder_name(scan)
            try:
                os.mkdir(name)
            except FileExistsError:
                # If there is a run in the folder, check if we are continuing an old run by matching the settings
                con_scan = self.check_run_history(scan)
                if not con_scan:
                    print(f"{name} folder present backing up folder to {name}_tmp")
                    # Remove old backups
                    try:
                        rmtree(f"{name}_tmp")
                    except FileNotFoundError:
                        pass

                    os.system(f"mv {name} {name}_tmp")
                    os.mkdir(name)

            os.chdir(name)

            make_and_change_into("QM_torsiondrive")

            # Start the torsion drive if psi4 else run native separate optimisations using g09
            self.start_torsiondrive(scan)
            # Get the scan results and load into the molecule
            os.chdir(self.home)


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
        self, molecule, weight_mm=True, step_size=0.02, error_tol=1e-5, x_tol=1e-5
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
        self.opt_coords = self.molecule.openmm_coordinates(input_type="qm")

        # constants
        self.k_b = constants.KB_KCAL_P_MOL_K
        self.phases = [0, constants.PI, 0, constants.PI]
        self.home = os.getcwd()

        # start the OpenMM system
        self.molecule.write_pdb()
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
            self.molecule.coords["traj"] = self.molecule.qm_scans[self.scan][1]
            self.scan_coords = self.molecule.openmm_coordinates(input_type="traj")
            self._create_rdkit_molecules(self.scan_coords)
            # Set up the fitting folders
            try:
                os.mkdir(f"SCAN_{self.scan[0]}_{self.scan[1]}")
            except FileExistsError:
                # back up the old scan data
                os.rename(
                    f"SCAN_{self.scan[0]}_{self.scan[1]}",
                    f"SCAN_{self.scan[0]}_{self.scan[1]}_backup",
                )
                os.mkdir(f"SCAN_{self.scan[0]}_{self.scan[1]}")

            # Make and move through the folders  and create the log file
            os.chdir(f"SCAN_{self.scan[0]}_{self.scan[1]}")
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
            self.optimiser_log.flush()
            # Get the optimisation method if it is a hybrid take the first option
            self.method = self.methods.get(self.molecule.opt_method.split("_")[0], None)
            error, opt_parameters = self.scipy_optimiser()
            self.optimiser_log.write(f"fitted parameters {opt_parameters}\n")
            self.optimiser_log.write(f"Fitted error {error}\n")
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
        if self.molecule.dihedral_types is None:
            self.molecule.dihedral_types = {}

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

        self.rmsd_atoms = []
        for coord in coordinates:
            rdkit_mol = deepcopy(self.molecule.rdkit_mol)
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
        self.molecule.write_pdb(name="openmm")
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
                self.molecule.coords["traj"] = self.molecule.qm_scans[self.scan][1]
                positions = self.molecule.openmm_coordinates(input_type="traj")

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

                self.molecule.save_to_ligand("scan.xyz")

            else:
                raise NotImplementedError(
                    "Invalid torsion engine. Please use torsiondrive or geometric"
                )

        # move back to the master folder
        os.chdir("../")

        return positions

    def single_point(self):
        """Take set of coordinates of a molecule and do a single point calculation; returns an array of the energies."""

        # reset the temp entry in the molecule
        self.molecule.coords["temp"] = self.molecule.coords["input"]
        # for each coordinate in the system we need to write a qm input file and get the single point energy
        try:
            rmtree("Single_points")
        except FileNotFoundError:
            pass

        make_and_change_into("Single_points")

        sp_energy = []

        for i, x in enumerate(self.scan_coords):
            make_and_change_into(f"SP_{i}")
            print(
                f"Doing single point calculations on new structures ... {i + 1}/{len(self.scan_coords)}"
            )
            # Change the positions of the molecule in the molecule array
            for y, coord in enumerate(x):
                for z, pos in enumerate(coord):
                    # Convert from nanometers in openmm to Angs in QM and store in the temp position in the molecule
                    self.qm_engine.molecule.coords["temp"][y][z] = (
                        pos * constants.NM_TO_ANGS
                    )

            # Write the new coordinate file and run the calculation
            result = self.qm_engine.generate_input(input_type="temp", energy=True)
            if result["success"] and self.qm_engine.__class__.__name__ == "Gaussian":
                coords, energy = self.qm_engine.optimised_structure()
                sp_energy.append(energy)

            elif result["success"] and self.qm_engine.__class__.__name__ == "PSI4":
                # Extract the energy and save to the array
                sp_energy.append(PSI4.get_energy())

            # Move back to the base directory
            os.chdir("../")

        # move out to the main folder
        os.chdir("../")

        return np.array(sp_energy)

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
