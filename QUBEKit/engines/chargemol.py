#!/usr/bin/env python3

import os
import subprocess as sp

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils.exceptions import ChargemolError
from QUBEKit.utils.helpers import append_to_log


class Chargemol(Engines):
    def __init__(self, molecule):

        super().__init__(molecule)

    def generate_input(self, execute=True):
        """Given a DDEC version (from the defaults), this function writes the job file for chargemol and executes it."""

        if (self.molecule.ddec_version != 6) and (self.molecule.ddec_version != 3):
            append_to_log(
                self.molecule.home,
                message="Invalid or unsupported DDEC version given, running with default version 6.",
                major=True,
                and_print=True,
            )
            self.molecule.ddec_version = 6

        # Write the charges job file.
        with open("job_control.txt", "w+") as charge_file:

            charge_file.write(
                f"<input filename>\n{self.molecule.name}.wfx\n</input filename>"
            )

            charge_file.write("\n\n<net charge>\n0.0\n</net charge>")

            charge_file.write(
                "\n\n<periodicity along A, B and C vectors>\n.false.\n.false.\n.false."
            )
            charge_file.write("\n</periodicity along A, B and C vectors>")

            charge_file.write(
                f"\n\n<atomic densities directory complete path>\n{self.molecule.chargemol}"
                f"/atomic_densities/"
            )
            charge_file.write("\n</atomic densities directory complete path>")

            charge_file.write(
                f"\n\n<charge type>\nDDEC{self.molecule.ddec_version}\n</charge type>"
            )

            charge_file.write("\n\n<compute BOs>\n.true.\n</compute BOs>")

            charge_file.write(
                "\n\n<print atomic densities>\n.true.\n</print atomic densities>"
            )

        if execute:
            # Export a variable to the environment that chargemol will use to work out the threads, must be a string
            os.environ["OMP_NUM_THREADS"] = str(self.molecule.threads)
            with open("log.txt", "w+") as log:
                control_path = (
                    "chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/"
                    "Chargemol_09_26_2017_linux_parallel job_control.txt"
                )
                try:
                    sp.run(
                        os.path.join(self.molecule.chargemol, control_path),
                        shell=True,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )

                except sp.CalledProcessError:
                    raise ChargemolError(
                        "Chargemol did not execute properly; check the output file for details."
                    )

                del os.environ["OMP_NUM_THREADS"]
