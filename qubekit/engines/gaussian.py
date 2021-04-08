#!/usr/bin/env python3

import os
import subprocess as sp

import numpy as np

from qubekit.engines.base_engine import Engines
from qubekit.utils import constants
from qubekit.utils.helpers import check_symmetry


class Gaussian(Engines):
    """
    Writes and executes input files for Gaussian09.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule):

        super().__init__(molecule)

        self.functional_dict = {
            "pbe": "PBEPBE",
            "wb97x-d": "wB97XD",
            "b3lyp-d3bj": "EmpiricalDispersion=GD3BJ B3LYP",
        }

        self.molecule.theory = self.functional_dict.get(
            self.molecule.theory.lower(), self.molecule.theory
        )

        self.convergence_dict = {
            "GAU": "",
            "GAU_TIGHT": "tight",
            "GAU_LOOSE": "loose",
            "GAU_VERYTIGHT": "verytight",
        }

    def generate_input(
        self,
        optimise=False,
        hessian=False,
        energy=False,
        density=False,
        restart=False,
        execute="g09",
    ):
        """
        Generates the relevant job file for Gaussian, then executes this job file.
        :param input_type: The set of coordinates in the molecule that should be used in the job
        :param optimise: Optimise the geometry of the molecule
        :param hessian: Calculate the hessian matrix
        :param energy: Calculate the single point energy
        :param density: Calculate the electron density
        :param restart: Restart from a check point file
        :param execute: Run the calculation after writing the input file
        :return: The exit status of the job if ran, True for normal false for not ran or error
        """

        if execute == "g16":
            print(
                "\nWe do not have the capability to test Gaussian 16; "
                "as a result, there may be some issues. "
                "Please let us know if any changes are needed through our Slack, or Github issues page.\n"
            )

        with open(f"gj_{self.molecule.name}.com", "w+") as input_file:

            input_file.write(
                f"%Mem={self.molecule.memory}GB\n%NProcShared={self.molecule.threads}\n%Chk=lig\n"
            )

            if self.molecule.excited_state:
                commands = f"# {self.molecule.theory}/{self.molecule.basis} "
                if self.molecule.use_pseudo:
                    commands += f" Pseudo=Read"

                commands += (
                    f" {self.molecule.excited_theory}=(Nstates={self.molecule.n_states}, "
                    f"Root={self.molecule.excited_root}) SCF=XQC "
                )

            else:
                commands = (
                    f"# {self.molecule.theory}/{self.molecule.basis} "
                    f"SCF=(XQC,MaxConventionalCycles={self.molecule.iterations}) nosymm "
                )

            # Adds the commands in groups. They MUST be in the right order because Gaussian.
            if optimise:
                commands += f"Opt=ModRedundant "

            if hessian:
                commands += "freq "

            if energy:
                commands += "SP "

            if density:
                commands += "density=current OUTPUT=WFX "
                if self.molecule.solvent:
                    commands += "SCRF=(IPCM,Read) "

            if restart:
                commands += "geom=check"

            commands += f"\n\n{self.molecule.name}\n\n{self.molecule.charge} {self.molecule.multiplicity}\n"

            input_file.write(commands)

            if not restart:
                # Add the atomic coordinates if we are not restarting from the chk file
                for atom_index, coord in enumerate(self.molecule.coordinates):
                    input_file.write(
                        f"{self.molecule.atoms[atom_index].atomic_symbol} "
                        f"{float(coord[0]): .10f} {float(coord[1]): .10f} {float(coord[2]): .10f}\n"
                    )

            # TODO finish this block
            if self.molecule.use_pseudo:
                input_file.write(f"\n{self.molecule.pseudo_potential_block}")

            if density and self.molecule.solvent:
                # Adds the epsilon and cavity params
                input_file.write(f"\n{self.molecule.dielectric} 0.0004")

            if density:
                # Specify the creation of the wavefunction file
                input_file.write(f"\n{self.molecule.name}.wfx")

            # Blank lines because Gaussian.
            input_file.write("\n\n\n\n")

        # execute should be either g09, g16 or False
        if execute:
            with open("log.txt", "w+") as log:
                sp.run(
                    f"{execute} < gj_{self.molecule.name}.com > gj_{self.molecule.name}.log",
                    shell=True,
                    stdout=log,
                    stderr=log,
                )

            # Now check the exit status of the job
            return self.check_for_errors()

        return {"success": False, "error": "Not run"}

    def check_for_errors(self):
        """
        Read the output file and check for normal termination and any errors.
        :return: A dictionary of the success status and any problems
        """

        with open(f"gj_{self.molecule.name}.log", "r") as log:
            for line in log:
                if "Normal termination of Gaussian" in line:
                    return {"success": True}

                elif "Problem with the distance matrix." in line:
                    return {"success": False, "error": "Distance matrix"}

                elif "Error termination in NtrErr" in line:
                    return {"success": False, "error": "FileIO"}

                elif "-- Number of steps exceeded" in line:
                    return {"success": False, "error": "Max iterations"}

            return {"success": False, "error": "Unknown"}

    def hessian(self):
        """Extract the Hessian matrix from the Gaussian fchk file."""

        # Make the fchk file first
        with open("formchck.log", "w+") as formlog:
            sp.run(
                "formchk lig.chk lig.fchk", shell=True, stdout=formlog, stderr=formlog
            )

        with open("lig.fchk", "r") as fchk:

            lines = fchk.readlines()

            # Improperly formatted Hessian (converted to square numpy array later)
            hessian_list = []
            start, end = None, None

            for count, line in enumerate(lines):
                if line.startswith("Cartesian Force Constants"):
                    start = count + 1
                if line.startswith("Nonadiabatic coupling"):
                    if end is None:
                        end = count
                if line.startswith("Dipole Moment"):
                    if end is None:
                        end = count

            if not start and end:
                raise EOFError("Cannot locate Hessian matrix in lig.fchk file.")

            conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS ** 2)
            for line in lines[start:end]:
                # Extend the list with the converted floats from the file, splitting on spaces and removing '\n' tags.
                hessian_list.extend(
                    [float(num) * conversion for num in line.strip("\n").split()]
                )

        hess_size = 3 * len(self.molecule.atoms)
        hessian = np.zeros((hess_size, hess_size))

        # Rewrite Hessian to full, symmetric 3N * 3N matrix rather than list with just the non-repeated values.
        m = 0
        for i in range(hess_size):
            for j in range(i + 1):
                hessian[i, j] = hessian_list[m]
                hessian[j, i] = hessian_list[m]
                m += 1

        check_symmetry(hessian)

        return hessian

    def optimised_structure(self):
        """
        Extract the optimised structure and energy from a fchk file
        :return molecule: The optimised array with the structure
        :return energy:  The SCF energy of the optimised structure
        """
        # Make the fchk file first
        with open("formchck.log", "w+") as formlog:
            sp.run(
                "formchk lig.chk lig.fchk", shell=True, stdout=formlog, stderr=formlog
            )

        with open("lig.fchk", "r") as fchk:
            lines = fchk.readlines()

        start, end, energy = None, None, None

        for count, line in enumerate(lines):
            if "Current cartesian coordinates" in line:
                start = count + 1
            elif "Number of symbols in" in line:
                if end is None:
                    end = count
            elif "Int Atom Types" in line:
                if end is None:
                    end = count - 1
            elif "Total Energy" in line:
                energy = float(line.split()[3])

        if any(val is None for val in [start, end, energy]):
            raise EOFError("Cannot locate optimised structure in file.")

        molecule = []
        # Now get the coords from the file
        for line in lines[start:end]:
            molecule.extend([float(coord) for coord in line.split()])

        molecule = (
            np.array(molecule).reshape((len(self.molecule.atoms), 3))
            * constants.BOHR_TO_ANGS
        )

        return molecule, energy

    def all_modes(self):
        """Extract the frequencies from the Gaussian log file."""

        with open(f"gj_{self.molecule.name}.log", "r") as gj_log_file:

            lines = gj_log_file.readlines()

            freqs = []

            # Stores indices of rows which will be used
            freq_positions = []
            for count, line in enumerate(lines):
                if line.startswith(" Frequencies"):
                    freq_positions.append(count)

            for pos in freq_positions:
                freqs.extend(float(num) for num in lines[pos].split()[2:])

        return np.array(freqs)

    @staticmethod
    def cleanup():
        """After a successful run, can be called to remove lig.chk, lig.fchk files."""

        files_to_remove = ["lig.chk", "lig.fchk", "log.txt"]
        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
