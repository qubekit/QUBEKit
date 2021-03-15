#!/usr/bin/env python3

import subprocess as sp

import numpy as np

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils.exceptions import PSI4Error
from QUBEKit.utils.helpers import append_to_log


class PSI4(Engines):
    """
    Writes and executes input files for psi4.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule):

        super().__init__(molecule)

        self.functional_dict = {"pbepbe": "PBE", "wb97xd": "wB97X-D"}
        # Search for functional in dict, if it's not there, just leave the theory as it is.
        self.molecule.theory = self.functional_dict.get(
            self.molecule.theory.lower(), self.molecule.theory
        )

        # Test if PSI4 is callable
        try:
            sp.run("psi4 -h", shell=True, check=True, stdout=sp.PIPE)
        except sp.CalledProcessError as exc:
            raise ModuleNotFoundError(
                "PSI4 not working. Please ensure PSI4 is installed and can be called with the command: psi4"
            ) from exc

        if self.molecule.geometric:
            try:
                sp.run("geometric-optimize -h", shell=True, check=True, stdout=sp.PIPE)
            except sp.CalledProcessError as exc:
                raise ModuleNotFoundError(
                    "Geometric not working. Please ensure geometric is installed and can be called "
                    "with the command: geometric-optimize"
                ) from exc

    # TODO add restart from log method
    def generate_input(
        self,
        optimise=False,
        hessian=False,
        density=False,
        energy=False,
        fchk=False,
        restart=False,
        execute=True,
    ):
        """
        Converts to psi4 input format to be run in psi4 without using geometric.
        :param optimise: Optimise the molecule to the desired convergence criterion within the iteration limit
        :param hessian: Calculate the hessian matrix
        :param density: Calculate the electron density
        :param energy: Calculate the single point energy of the molecule
        :param fchk: Write out a gaussian style Fchk file
        :param restart: Restart the calculation from a log point (required but unused to match g09's generate_input())
        :param execute: Run the desired Psi4 job
        :return: The completion status of the job True if successful False if not run or failed
        """

        setters = ""
        tasks = ""

        if energy:
            append_to_log(self.molecule.home, "Writing psi4 energy calculation input")
            tasks += f"\nenergy('{self.molecule.theory}')"

        if optimise:
            append_to_log(
                self.molecule.home, "Writing PSI4 optimisation input", "minor"
            )
            setters += f" g_convergence {self.molecule.convergence}\n GEOM_MAXITER {self.molecule.iterations}\n"
            tasks += f"\noptimize('{self.molecule.theory.lower()}')"

        if hessian:
            append_to_log(
                self.molecule.home,
                "Writing PSI4 Hessian matrix calculation input",
            )
            setters += " hessian_write on\n"

            tasks += f"\nenergy, wfn = frequency('{self.molecule.theory.lower()}', return_wfn=True)"

            tasks += "\nwfn.hessian().print_out()\n\n"

        if density:
            raise NotImplementedError(
                "Due to PSI4 requiring a box size which cannot be automatically generated, "
                "PSI4 cannot currently be used for density calculations. Please use Gaussian "
                "instead."
            )

        if fchk:
            append_to_log(
                self.molecule.home, "Writing PSI4 input file to generate fchk file"
            )
            tasks += f"\ngrad, wfn = gradient('{self.molecule.theory.lower()}', return_wfn=True)"
            tasks += "\nfchk_writer = psi4.core.FCHKWriter(wfn)"
            tasks += f'\nfchk_writer.write("{self.molecule.name}_psi4.fchk")\n'

        setters += "}\n"

        if not execute:
            setters += f"set_num_threads({self.molecule.threads})\n"

        # input.dat is the PSI4 input file.
        with open("input.dat", "w+") as input_file:
            # opening tag is always writen
            input_file.write(
                f"memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n"
                f"{self.molecule.charge} {self.molecule.multiplicity} \n"
            )
            # molecule is always printed
            for i, atom in enumerate(self.molecule.coordinates):
                input_file.write(
                    f" {self.molecule.atoms[i].atomic_symbol}    "
                    f"{float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f} \n"
                )

            input_file.write(
                f" units angstrom\n no_reorient\n}}\n\nset {{\n basis {self.molecule.basis}\n"
            )

            input_file.write(setters)
            input_file.write(tasks)

        if execute:
            with open("log.txt", "w+") as log:
                try:
                    sp.run(
                        f"psi4 input.dat -n {self.molecule.threads}",
                        shell=True,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
                except sp.CalledProcessError as exc:
                    raise PSI4Error(
                        "PSI4 did not execute successfully check log file for details."
                    ) from exc

            # Now check the exit status of the job
            return self.check_for_errors()

        else:
            return {"success": False, "error": "Not run"}

    def check_for_errors(self):
        """
        Read the output file from the job and check for normal termination and any errors
        :return: A dictionary of the success status and any errors.
        """

        with open("output.dat", "r") as log:
            for line in log:
                if "*** Psi4 exiting successfully." in line:
                    return {"success": True}

                elif "*** Psi4 encountered an error." in line:
                    return {"success": False, "error": "Not known"}

            return {"success": False, "error": "Segfault"}

    def optimised_structure(self):
        """
        Parses the final optimised structure from the output.dat file (from psi4) to a numpy array.
        Also returns the energy of the optimized structure.
        """

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # output.dat is the psi4 output file.
        with open("output.dat", "r") as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if "==> Geometry" in line:
                    geo_pos_list.append(count)

                elif "**** Optimization is complete!" in line:
                    opt_pos = count
                    opt_steps = int(line.split()[5])

            if not (opt_pos and opt_steps):
                raise EOFError(
                    "According to the output.dat file, optimisation has not completed."
                )

            # now get the final opt_energy
            opt_energy = float(lines[opt_pos + opt_steps + 7].split()[1])

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            opt_struct = []

            for row in range(len(self.molecule.atoms)):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = []
                for indx in range(3):
                    struct_row.append(
                        float(lines[start_of_vals + row].split()[indx + 1])
                    )

                opt_struct.append(struct_row)

        return np.array(opt_struct), opt_energy

    @staticmethod
    def get_energy():
        """Get the energy of a single point calculation."""

        # open the psi4 log file
        with open("output.dat", "r") as log:
            for line in log:
                if "Total Energy =" in line:
                    return float(line.split()[3])

        raise EOFError("Cannot find energy in output.dat file.")

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # output.dat is the psi4 output file.
        with open("output.dat", "r") as file:
            lines = file.readlines()
            for count, line in enumerate(lines):
                if "post-proj  all modes" in line:
                    start_of_vals = count
                    break
            else:
                raise EOFError("Cannot locate modes in output.dat file.")

            # Barring the first (and sometimes last) line, dat file has 6 values per row.
            end_of_vals = start_of_vals + (3 * len(self.molecule.atoms)) // 6

            structures = lines[start_of_vals][24:].replace("'", "").split()
            structures = structures[6:]

            for row in range(1, end_of_vals - start_of_vals):
                # Remove double strings and weird formatting.
                structures += (
                    lines[start_of_vals + row].replace("'", "").replace("]", "").split()
                )

            all_modes = [float(val) for val in structures]

            return np.array(all_modes)

    def geo_gradient(self, threads=False, execute=True):
        """
        Write the psi4 style input file to get the gradient for geometric
        and run geometric optimisation.
        """

        molecule = self.molecule.coordinates

        with open(f"{self.molecule.name}.psi4in", "w+") as file:

            file.write(
                f"memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n {self.molecule.charge} "
                f"{self.molecule.multiplicity} \n"
            )

            for i, atom in enumerate(molecule):
                file.write(
                    f"  {self.molecule.atoms[i].atomic_symbol:2}    "
                    f"{float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f}\n"
                )

            file.write(
                f" units angstrom\n no_reorient\n}}\nset basis {self.molecule.basis}\n"
            )

            if threads:
                file.write(f"set_num_threads({self.molecule.threads})")

            file.write(f"\n\ngradient('{self.molecule.theory}')\n")

        if execute:
            with open("log.txt", "w+") as log:
                sp.run(
                    f"geometric-optimize --psi4 {self.molecule.name}.psi4in {self.molecule.constraints_file} "
                    f"--nt {self.molecule.threads}",
                    shell=True,
                    stdout=log,
                    stderr=log,
                )
