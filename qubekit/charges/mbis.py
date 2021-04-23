"""
An interface to MBIS via psi4.
"""

from pydantic import Field
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.charges.base import ChargeBase
from qubekit.charges.solvent_settings import SolventPsi4
from qubekit.engines import call_qcengine
from qubekit.molecules import Dipole, Ligand, Quadrupole
from qubekit.utils.datastructures import LocalResource


class MBISCharges(ChargeBase):

    type: Literal["MBISCharges"] = "MBISCharges"
    solvent_settings: SolventPsi4 = Field(
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
        p4 = which_import(
            "psi4",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install psi4 -c psi4/label/dev`.",
        )
        return p4

    def _run(self, molecule: "Ligand", local_options: LocalResource) -> "Ligand":
        """
        The main run method which generates a density using psi4 and stores the partitioned MBIS AIM reference values.
        """
        # now we need to build the keywords for the solvent
        extras = dict(
            pcm=True,
            pcm__input=self.solvent_settings.format_keywords(),
            scf_properties=["MBIS_CHARGES"],
        )
        result = call_qcengine(
            molecule=molecule,
            driver="energy",
            qc_spec=self._get_qc_options(),
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
