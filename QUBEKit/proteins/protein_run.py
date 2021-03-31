#!/usr/bin/env python3

import argparse
import os
import sys
from functools import partial

from QUBEKit.lennard_jones import LennardJones612
from QUBEKit.molecules import Protein
from QUBEKit.parametrisation import XMLProtein
from QUBEKit.proteins.protein_tools import get_water, pdb_reformat, qube_general
from QUBEKit.utils.file_handling import ExtractChargeData

printf = partial(print, flush=True)


def main():
    """
    This script is used to prepare proteins with the QUBE FF
    1) prepare the protein for onetep using --setup which prints an xyz of the system
    2) after the onetep calculation bring back the ddec.onetep file and parametrise the system with --build
    this must be the same pdb used in setup as the atom order must be retained.
    """

    # Setup the action classes
    class SetupAction(argparse.Action):
        """This class is called when we setup a new protein."""

        def __call__(self, pars, namespace, values, option_string=None):

            printf("starting protein prep, reading pdb file...")
            protein = Protein(values)
            printf(f"{len(protein.residues)} residues found!")
            # TODO find the magic numbers for the box for onetep
            protein.to_file(file_name="protein.xyz")
            printf(f"protein.xyz file made for ONETEP\n Run this file")
            sys.exit()

    class BuildAction(argparse.Action):
        """This class handles the building of the protein xml and pdb files."""

        def __call__(self, pars, namespace, values, option_string=None):

            pro = Protein(values)
            # print the QUBE general FF to use in the parametrisation
            qube_general()
            # now we want to add the connections and parametrise the protein
            XMLProtein(pro)

            # finally we need the non-bonded parameters from onetep
            # TODO should we also have the ability to get DDEC6 charges from the cube file?
            pro.charges_engine = "onetep"
            pro.density_engine = "onetep"

            ExtractChargeData.read_files(pro, pro.home, "onetep")
            LennardJones612(pro).calculate_non_bonded_force()

            # Write out the final parameters
            printf("Writing pdb file with connections...")
            pro.write_pdb(name=f"QUBE_pro_{pro.name}")
            printf("Writing XML file for the system...")
            pro.write_parameters(file_name=f"QUBE_pro_{pro.name}")
            # now remove the qube general file
            os.remove("QUBE_general_pi.xml")
            printf("Done")
            sys.exit()

    class WaterAction(argparse.Action):
        """This class builds the water models requested"""

        def __call__(self, pars, namespace, values, option_string=None):
            """This function is executed when water is called."""
            get_water(values)
            sys.exit()

    class ConvertAction(argparse.Action):
        """This class converts the names in a qube taj file to match the reference."""

        def __call__(self, pars, namespace, values, option_string=None):
            """This function is executed when convert is called."""
            reference, target = values
            printf(reference, target)
            printf(f"Rewriting input: {target} to match: {reference}...")
            pdb_reformat(reference, target)
            printf("Done output made: QUBE_traj.pdb")
            sys.exit()

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        prog="QUBEKit-pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="QUBEKit-pro is an extension to QUBEKit that allows the preparation"
        "of amber prepared proteins with the QUBE forcefield and creates force"
        "field xml files for OpenMM simulation.",
    )
    parser.add_argument(
        "-setup",
        "--setup",
        action=SetupAction,
        help="Enter the name of the amber prepared protein file to convert to xyz for onetep.",
    )

    parser.add_argument(
        "-build",
        "--build_simulation_files",
        action=BuildAction,
        help="Enter the name of the amber prepared protein file to generate QUBE pdb and xml files.",
    )

    parser.add_argument(
        "-water",
        "--water",
        nargs="?",
        default="help",
        action=WaterAction,
        choices=[
            "help",
            "tip3p",
            "spce",
            "tip4p",
            "tip4pew",
            "tip5p",
            "tip3pfb",
            "tip4pfb",
            "tip4p-d",
            "opc",
        ],
        help="Enter the name of the water model you would like to use.",
    )

    parser.add_argument(
        "-convert",
        "--convert",
        nargs=2,
        action=ConvertAction,
        help="Enter the reference followed by the qube traj file you want to re format.",
    )

    parser.parse_args()


if __name__ == "__main__":
    main()
