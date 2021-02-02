"""
A method to parameterise a molecule using gromacs input files.
"""

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.ligand import Ligand

from simtk.openmm import System, app
from typing import Optional, List


class Gromacs(Parametrisation):
    """
    Parameterise a molecule using gromacs input files via openmm.
    """

    def __init__(self, fftype="gromacs"):
        super().__init__(fftype)

    def build_system(self, molecule: Ligand, input_files: Optional[List[str]] = None) -> System:
        """
        Using gromacs input files parameterise a molecule.
        """
        # get the topfile
        top_file = None
        if input_files is not None:
            for file in input_files:
                if file.endswith(".top"):
                    top_file = file
                    break
        # if it was not given try and guess it
        top_file = top_file or f"{molecule.name}.top"

        top = app.GromacsTopFile(top_file)
        system = top.createSystem()
        return system
