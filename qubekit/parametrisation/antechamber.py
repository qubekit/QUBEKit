import os
import shutil
import subprocess as sp
from tempfile import TemporaryDirectory
from typing import List, Optional

from qcelemental.util import which
from simtk.openmm import System, app
from typing_extensions import Literal

from qubekit.molecules import Ligand
from qubekit.parametrisation.base_parametrisation import Parametrisation


class AnteChamber(Parametrisation):
    """
    Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    type: Literal["AnteChamber"] = "AnteChamber"
    force_field: Literal["gaff", "gaff2"] = "gaff2"

    def start_message(self, **kwargs) -> str:
        return f"Parametrising molecule with {self.force_field}."

    @classmethod
    def is_available(cls) -> bool:
        ate = which(
            "antechamber",
            raise_error=True,
            return_bool=True,
            raise_msg="Please install ambertools using `conda install ambertools -c conda-forge.`",
        )
        return ate

    @classmethod
    def _improper_torsion_ordering(cls) -> str:
        return "amber"

    def _build_system(
        self, molecule: Ligand, input_files: Optional[List[str]] = None
    ) -> System:
        """
        Build a system using the amber prmtop files, first we must use antechamber to prep the molecule.
        """
        prmtoop_file = self._get_prmtop(molecule=molecule)
        prmtop = app.AmberPrmtopFile(prmtoop_file)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
        return system

    def _get_prmtop(self, molecule: Ligand) -> str:
        """Method to run Antechamber, parmchk2 and tleap."""

        # file paths when moving in and out of temp locations
        cwd = os.getcwd()
        mol2 = os.path.abspath(f"{molecule.name}.mol2")
        frcmod_file = os.path.abspath(f"{molecule.name}.frcmod")
        prmtop_file = os.path.abspath(f"{molecule.name}.prmtop")
        inpcrd_file = os.path.abspath(f"{molecule.name}.inpcrd")
        ant_log = os.path.abspath("Antechamber.log")

        # Call Antechamber
        # Do this in a temp directory as it produces a lot of files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            molecule.to_file(file_name="in.sdf")

            # Call Antechamber
            cmd = f"antechamber -i in.sdf -fi sdf -o out.mol2 -fo mol2 -s 2 -m {molecule.multiplicity} -c bcc -nc {molecule.charge} -pf yes"

            with open("ante_log.txt", "w+") as log:
                sp.run(cmd, shell=True, stdout=log, stderr=log)

            # Ensure command worked
            try:
                # Copy the gaff mol2 and antechamber file back
                shutil.copy("ante_log.txt", cwd)
                shutil.copy("out.mol2", mol2)
            except FileNotFoundError:
                os.chdir(cwd)
                raise FileNotFoundError(
                    "Antechamber could not convert this file type; is it a valid pdb?"
                )

            os.chdir(cwd)

        # Work in temp directory due to the amount of files made by antechamber
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(mol2, "out.mol2")

            # Run parmchk
            with open("Antechamber.log", "a") as log:
                sp.run(
                    f"parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s {self.force_field}",
                    shell=True,
                    stdout=log,
                    stderr=log,
                )

            # Ensure command worked
            if not os.path.exists("out.frcmod"):
                raise FileNotFoundError("out.frcmod not found parmchk2 failed!")

            # Now get the files back from the temp folder
            shutil.copy("out.mol2", mol2)
            shutil.copy("out.frcmod", frcmod_file)
            shutil.copy("Antechamber.log", ant_log)

        # Now we need to run tleap to get the prmtop and inpcrd files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            shutil.copy(mol2, "in.mol2")
            shutil.copy(frcmod_file, "in.frcmod")
            shutil.copy(ant_log, "Antechamber.log")

            # make tleap command file
            with open("tleap_commands", "w+") as tleap:
                tleap.write(
                    """source oldff/leaprc.ff99SB
                               source leaprc.gaff
                               LIG = loadmol2 in.mol2
                               check LIG
                               loadamberparams in.frcmod
                               saveamberparm LIG out.prmtop out.inpcrd
                               quit"""
                )

            # Now run tleap
            with open("Antechamber.log", "a") as log:
                sp.run("tleap -f tleap_commands", shell=True, stdout=log, stderr=log)

            # Check results present
            if not os.path.exists("out.prmtop") or not os.path.exists("out.inpcrd"):
                raise FileNotFoundError(
                    "Neither out.prmtop nor out.inpcrd found; tleap failed!"
                )

            shutil.copy("Antechamber.log", ant_log)
            shutil.copy("out.prmtop", prmtop_file)
            shutil.copy("out.inpcrd", inpcrd_file)
            os.chdir(cwd)

        # Now give the file names to parametrisation method
        prmtop = f"{molecule.name}.prmtop"
        return prmtop
