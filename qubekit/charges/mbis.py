"""
An interface to MBIS via psi4.
"""

from typing import Any, Dict, Optional

from pydantic import Field
from qcelemental.util import parse_version, safe_version, which, which_import
from qcengine.util import popen
from typing_extensions import Literal

from qubekit.charges.base import ChargeBase
from qubekit.charges.solvent_settings import SolventPsi4
from qubekit.engines import call_qcengine
from qubekit.molecules import Dipole, Ligand, Quadrupole
from qubekit.utils.datastructures import LocalResource, QCOptions
from qubekit.utils.exceptions import SpecificationError


class MBISCharges(ChargeBase):
    type: Literal["MBISCharges"] = "MBISCharges"
    solvent_settings: Optional[SolventPsi4] = Field(
        SolventPsi4(units="au", medium_Solvent="chloroform"),
        description="The engine that should be used to generate the reference density to perform the AIM analysis on.",
    )
    program: Literal["psi4"] = "psi4"

    def start_message(self, **kwargs) -> str:
        return "Calculating charges using MBIS via psi4."

    @classmethod
    def is_available(cls) -> bool:
        """
        The MBIS option is only available via new psi4 so make sure it is installed.
        """
        # check installed
        psi4 = which_import(
            "psi4",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install psi4 -c psi4`.",
        )
        # now check the version meets the minimum requirement
        which_psi4 = which("psi4")
        with popen([which_psi4, "--version"]) as exc:
            exc["proc"].wait(timeout=30)
        version = parse_version(safe_version(exc["stdout"].split()[-1]))
        if version <= parse_version("1.4a1"):
            raise SpecificationError(
                f"The version of psi4 installed is {version} and needs to be 1.4 or newer please update it to continue."
            )
        return psi4

    def _gas_calculation_settings(self) -> Dict[str, Any]:
        return {"scf_properties": ["MBIS_CHARGES"]}

    def _execute(
        self, molecule: "Ligand", local_options: LocalResource, qc_spec: QCOptions
    ) -> "Ligand":
        """
        The main run method which generates a density using psi4 and stores the partitioned MBIS AIM reference values.
        """
        # now we need to build the keywords for the solvent
        extras = self._get_calculation_settings()
        result = call_qcengine(
            molecule=molecule,
            driver="energy",
            qc_spec=qc_spec,
            local_options=local_options,
            extras=extras,
        )
        # pick out the MBIS data and store into the molecule.
        qcvars = result.extras["qcvars"]
        charges = qcvars["MBIS CHARGES"]
        dipoles = qcvars["MBIS DIPOLES"]
        quadrupoles = qcvars["MBIS QUADRUPOLES"]
        # not used yet
        # octupoles = qcvars["MBIS OCTUPOLES"]
        volumes = qcvars["MBIS RADIAL MOMENTS <R^3>"]
        # loop over the atoms and store the data
        for i in range(molecule.n_atoms):
            atom = molecule.atoms[i]
            atom.aim.charge = charges[i][0]
            atom.aim.volume = volumes[i][0]
            dipole = Dipole(x=dipoles[i][0], y=dipoles[i][1], z=dipoles[i][2])
            atom.dipole = dipole
            trace = quadrupoles[i][0][0] + quadrupoles[i][1][1] + quadrupoles[i][2][2]
            trace /= 3
            # make sure we have the traceless quad tensor
            quad = Quadrupole(
                q_xx=quadrupoles[i][0][0] - trace,
                q_xy=quadrupoles[i][0][1],
                q_xz=quadrupoles[i][0][2],
                q_yy=quadrupoles[i][1][1] - trace,
                q_zz=quadrupoles[i][2][2] - trace,
                q_yz=quadrupoles[i][1][2],
            )
            atom.quadrupole = quad

        return molecule
