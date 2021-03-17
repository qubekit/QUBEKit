"""
A custom gaussian harness for QCEngine which should be registered with qcengine.
"""
from typing import Any, Dict, List, Optional, Tuple

from qcelemental.models import AtomicInput, AtomicResult
from qcelemental.util import which
from qcengine import register_program
from qcengine.exceptions import UnknownError
from qcengine.programs import ProgramHarness
from qcengine.util import execute

from QUBEKit.utils import constants
from QUBEKit.utils.file_handling import get_data


class GaussianHarness(ProgramHarness):
    """
    A very minimal Gaussian wrapper for optimisations and torsiondrives.
    Note a lot of key information is not extracted need for QCArchive.
    """

    name = "gaussian"
    scratch = True
    thread_safe = True
    thread_parallel = True
    node_parallel = False
    managed_memory = True

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        """
        Check if gaussian can be found via the command line, check for g09 and g16.
        """
        gaussian09 = which("g09", return_bool=True)
        gaussian16 = which("g16", return_bool=True)
        return gaussian09 or gaussian16

    def get_version(self) -> str:
        """
        Work out which version of gaussian we should be running.
        """
        if which("g09", return_bool=True, raise_error=False):
            return "g09"
        else:
            which(
                "g16",
                return_bool=True,
                raise_error=True,
                raise_msg="Gaussian 09/16 can not be found make sure they are available.",
            )
            return "g16"

    def compute(self, input_data: AtomicInput, config: "TaskConfig") -> AtomicResult:
        """
        Run the compute job via the gaussian CLI.
        """
        # we always use an internal template
        job_inputs = self.build_input(input_model=input_data, config=config)
        exe_success, proc = self.execute(job_inputs)
        if exe_success:
            result = self.parse_output(proc["outfiles"], input_data)
            return result
        else:
            raise UnknownError(proc["stderr"])

    def execute(
        self,
        inputs: Dict[str, Any],
        extra_outfiles: Optional[List[str]] = None,
        extra_commands: Optional[List[str]] = None,
        scratch_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the gaussian single point job and fchk conversion
        """
        # collect together inputs
        infiles = inputs["infiles"]
        outfiles = ["gaussian.log", "lig.chk"]
        gaussian_version = self.get_version()
        commands = [gaussian_version, "gaussian.com"]
        scratch_directory = inputs["scratch_directory"]

        # remember before formatting lig.chk is binary, run calculation
        exe_success, proc = execute(
            command=commands,
            infiles=infiles,
            outfiles=outfiles,
            scratch_directory=scratch_directory,
            as_binary=["lig.chk"],
        )
        if exe_success:
            # now we need to run the conversion of the chk file
            commands = ["formchk", "lig.chk", "lig.fchk"]
            infiles = {"lig.chk": proc["outfiles"]["lig.chk"]}
            chk_success, proc_chk = execute(
                command=commands,
                infiles=infiles,
                outfiles=["lig.fchk"],
                scratch_directory=scratch_directory,
                as_binary=["lig.chk"],
            )
            # now we want to put this out file into the main proc
            proc["outfiles"]["lig.fchk"] = proc_chk["outfiles"]["lig.fchk"]
            # we can dump the chk file
            del proc["outfiles"]["lig.chk"]

        return exe_success, proc

    @staticmethod
    def functional_converter(method: str) -> str:
        """
        Convert the given function to the correct format for gaussian.
        """
        functionals = {"pbe": "PBEPBE", "wb97x-d": "wB97XD"}

        # check for dispersion
        if "-d3" in method:
            theory = method.split("-d3")[0]
            dispersion = "GD3"
            if "bj" in method:
                dispersion += "BJ"
        else:
            theory = method
            dispersion = None
        # convert the theory
        theory = functionals.get(theory, theory)
        if dispersion is not None:
            return f"EmpiricalDispersion={dispersion.upper()} {theory}"
        return theory

    @staticmethod
    def driver_conversion(driver: str) -> str:
        """
        Convert the qcengine driver enum to the specific keyword for gaussian.
        """
        drivers = {"energy": "SP", "gradient": "Force=NoStep", "hessian": "FREQ"}
        return drivers[driver.lower()]

    def build_input(
        self,
        input_model: AtomicInput,
        config: "TaskConfig",
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use the template files stored in QUBEKit to build a gaussian input file for the given driver.
        """
        import os

        from jinja2 import Template

        template_file = get_data(os.path.join("templates", "gaussian.com"))
        with open(template_file) as file:
            template = Template(file.read())

        template_data = {
            "memory": int(config.memory),
            "threads": config.ncores,
            "driver": self.driver_conversion(driver=input_model.driver),
            "title": "gaussian job",
        }
        molecule = input_model.molecule
        spec = input_model.model
        theory = self.functional_converter(method=spec.method)
        template_data["theory"] = theory
        template_data["basis"] = spec.basis
        template_data["charge"] = int(molecule.molecular_charge)
        template_data["multiplicity"] = molecule.molecular_multiplicity
        template_data["scf_maxiter"] = input_model.extras.get("maxiter", 300)
        # now we need to build the coords data
        data = []
        for i, symbol in enumerate(molecule.symbols):
            # we must convert the atomic input back to angstroms
            data.append((symbol, molecule.geometry[i] * constants.BOHR_TO_ANGS))
        template_data["data"] = data

        rendered_template = template.render(**template_data)
        return {
            "infiles": {"gaussian.com": rendered_template},
            "scratch_directory": config.scratch_directory,
            "input_result": input_model.copy(deep=True),
        }

    @staticmethod
    def check_convergence(logfile: str) -> None:
        """
        Check the gaussian log file to make sure we have normal termination.
        """
        if "Normal termination of Gaussian" not in logfile:
            # raise an error with the log file as output
            raise UnknownError(message=logfile)

    def parse_output(
        self, outfiles: Dict[str, str], input_model: AtomicInput
    ) -> AtomicResult:
        """
        For the set of output files parse them to extract as much info as possible and return the atomic result.
        From the fchk file we get the energy and hessian, the gradient is taken from the log file.
        """
        properties = {}
        # make sure we got vaild exit status
        self.check_convergence(logfile=outfiles["gaussian.log"])
        version = self.parse_version(logfile=outfiles["gaussian.log"])
        # build the main data dict
        output_data = input_model.dict()
        provenance = {"version": version, "creator": "gaussian", "routine": "CLI"}
        # collect the total energy from the fchk file
        logfile = outfiles["lig.fchk"]
        for line in logfile.split("\n"):
            if "Total Energy" in line:
                energy = float(line.split()[3])
                properties["return_energy"] = energy
                properties["scf_total_energy"] = energy
                if input_model.driver == "energy":
                    output_data["return_result"] = energy
        if input_model.driver == "gradient":
            # now we need to parse out the forces
            gradient = self.parse_gradient(fchfile=outfiles["lig.fchk"])
            output_data["return_result"] = gradient
        elif input_model.driver == "hessian":
            hessian = self.parse_hessian(fchkfile=outfiles["lig.fchk"])
            output_data["return_result"] = hessian

        output_data["properties"] = properties
        output_data["schema_name"] = "qcschema_output"
        output_data["stdout"] = outfiles["gaussian.log"]
        output_data["success"] = True
        output_data["provenance"] = provenance
        return AtomicResult(**output_data)

    @staticmethod
    def parse_version(logfile: str) -> str:
        """
        Try and parse the gaussian version from the logfile.
        """
        # the version is printed after the first ***** line
        lines = logfile.split("\n")
        star_line = 0
        for i, line in enumerate(lines):
            if "******************************************" in line:
                star_line = i
                break
        # now extract the line
        version = lines[star_line + 1].strip()
        return version

    @staticmethod
    def parse_gradient(fchfile: str) -> List[float]:
        """
        Parse the cartesian gradient from a gaussian fchk file and return them as a flat list.
        """
        gradient = []
        grad_found = False
        for line in fchfile.split("\n"):
            if "Cartesian Gradient" in line:
                grad_found = True
            elif grad_found:
                if "Cartesian Force Constants" in line or "Dipole Moment" in line:
                    grad_found = False
                else:
                    gradient.extend([float(grad) for grad in line.split()])
        return gradient

    @staticmethod
    def parse_hessian(fchkfile: str) -> List[float]:
        """
        Parse the hessian from the fchk file and return as a flat list in atomic units.
        Note gaussian gives us only half of the matrix so we have to fix this.
        #TODO the reshaping or the array could be fragile is there a better way?
        """
        import numpy as np

        hessian = []
        hessian_found = False
        for line in fchkfile.split("\n"):
            if line.startswith("Cartesian Force Constants"):
                hessian_found = True
            elif hessian_found:
                if line.startswith("Nonadiabatic coupling") or line.startswith(
                    "Dipole Moment"
                ):
                    hessian_found = False
                else:
                    hessian.extend([float(hess) for hess in line.split()])
        # now we can calculate the size of the hessian
        # (2 * triangle length) + 0.25 = (x+0.5)^2
        hess_size = int(np.sqrt(2 * len(hessian) + 0.25) - 0.5)
        full_hessian = np.zeros((hess_size, hess_size))
        m = 0
        for i in range(hess_size):
            for j in range(i + 1):
                full_hessian[i, j] = hessian[m]
                full_hessian[j, i] = hessian[m]
                m += 1

        return full_hessian.flatten().tolist()


# register the gaussian harness this must be done so gaussian can be used!
register_program(GaussianHarness())
