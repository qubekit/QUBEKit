import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from xml.dom.minidom import parseString

import click

import qubekit
from qubekit.utils import constants
from qubekit.workflow import WorkFlowResult

if TYPE_CHECKING:
    from qubekit.molecules import Atom, Ligand


elements = ["H", "C", "N", "O", "X", "Cl", "S", "F", "Br", "P", "I", "B", "Si", "AB"]
parameters_to_fit = click.Choice(
    elements,
    case_sensitive=True,
)


@click.command()
@click.argument("filename", type=click.STRING)
@click.option(
    "-p",
    "--parameters",
    type=parameters_to_fit,
    help="The elements whose Rfree is to be optimised, if not provided all will be fit.",
    multiple=True,
)
@click.option(
    "-n",
    "--no-targets",
    is_flag=True,
    show_default=True,
    default=False,
    help="If the xmls should be combined with no optimisation targets, this option is useful for Forcebalance single point evaluations.",
)
def combine(
    filename: str, parameters: Optional[List[str]] = None, no_targets: bool = False
):
    """
    Combine a list of molecules together and create a single master XML force field file.
    """
    if no_targets and parameters:
        raise click.ClickException(
            "The options parameters and no-targets are mutually exclusive."
        )

    molecules, rfrees = _find_molecules_and_rfrees()
    if not parameters and not no_targets:
        # fit everything
        parameters = elements
    elif no_targets:
        parameters = []

    xml_data = _combine_molecules(
        molecules=molecules, rfree_data=rfrees, parameters=parameters
    ).getroot()
    messy = ET.tostring(xml_data, "utf-8")

    pretty_xml = parseString(messy).toprettyxml(indent="")

    with open(filename, "w") as xml_doc:
        xml_doc.write(pretty_xml)


def _combine_molecules(
    molecules: List["Ligand"],
    parameters: List[str],
    rfree_data: Dict[str, Dict[str, Union[str, float]]],
) -> ET.ElementTree:
    """
    Main worker function used to combine  the molecule force fields.
    """
    fit_ab = False
    # if alpha and beta should be fit
    if "AB" in parameters:
        fit_ab = True

    rfree_codes = set()  # keep track of all rfree codes used by these molecules

    root = ET.Element("ForceField")
    ET.SubElement(
        root,
        "QUBEKit",
        attrib={
            "Version": qubekit.__version__,
            "Date": datetime.now().strftime("%Y_%m_%d"),
        },
    )
    AtomTypes = ET.SubElement(root, "AtomTypes")
    Residues = ET.SubElement(root, "Residues")

    force_by_type = {}

    increment = 0

    # loop over molecules and create forces as they are found
    for molecule in molecules:

        resiude = ET.SubElement(Residues, "Residue", name=molecule.name)

        for atom in molecule.atoms:
            atom_type = f"QUBE_{atom.atom_index + increment}"
            ET.SubElement(
                AtomTypes,
                "Type",
                attrib={
                    "name": atom_type,
                    "class": str(atom.atom_index + increment),
                    "element": atom.atomic_symbol,
                    "mass": str(atom.atomic_mass),
                },
            )

            ET.SubElement(
                resiude,
                "Atom",
                attrib={"name": str(atom.atom_index + increment), "type": atom_type},
            )

        for (
            i,
            site,
        ) in enumerate(molecule.extra_sites, start=1 + increment):
            site_name = f"v-site{i}"
            site_class = f"X{i}"
            ET.SubElement(
                AtomTypes,
                "Type",
                attrib={"name": site_name, "class": site_class, "mass": "0"},
            )
            ET.SubElement(
                resiude, "Atom", attrib={"name": site_class, "type": site_name}
            )

        if molecule.BondForce.openmm_group() not in force_by_type:
            new_bond_force = ET.SubElement(
                root,
                molecule.BondForce.openmm_group(),
                attrib=molecule.BondForce.xml_data(),
            )
            # add to dict
            force_by_type[molecule.BondForce.openmm_group()] = new_bond_force

        BondForce = force_by_type[molecule.BondForce.openmm_group()]
        for parameter in molecule.BondForce:
            ET.SubElement(
                BondForce,
                parameter.openmm_type(),
                attrib=_update_increment(
                    force_data=parameter.xml_data(), increment=increment
                ),
            )
            ET.SubElement(
                resiude,
                "Bond",
                attrib={"from": str(parameter.atoms[0]), "to": str(parameter.atoms[1])},
            )

        if molecule.AngleForce.openmm_group() not in force_by_type:
            new_angle_force = ET.SubElement(
                root,
                molecule.AngleForce.openmm_group(),
                attrib=molecule.AngleForce.xml_data(),
            )
            force_by_type[molecule.AngleForce.openmm_group()] = new_angle_force

        AngleForce = force_by_type[molecule.AngleForce.openmm_group()]
        for parameter in molecule.AngleForce:
            ET.SubElement(
                AngleForce,
                parameter.openmm_type(),
                attrib=_update_increment(
                    force_data=parameter.xml_data(), increment=increment
                ),
            )

        if molecule.TorsionForce.openmm_group() not in force_by_type:
            new_torsion_force = ET.SubElement(
                root,
                molecule.TorsionForce.openmm_group(),
                attrib=molecule.TorsionForce.xml_data(),
            )
            force_by_type[molecule.TorsionForce.openmm_group()] = new_torsion_force

        TorsionForce = force_by_type[molecule.TorsionForce.openmm_group()]

        for parameter in molecule.TorsionForce:
            ET.SubElement(
                TorsionForce,
                parameter.openmm_type(),
                attrib=_update_increment(
                    force_data=parameter.xml_data(), increment=increment
                ),
            )
        for parameter in molecule.ImproperTorsionForce:
            ET.SubElement(
                TorsionForce,
                parameter.openmm_type(),
                attrib=_update_increment(
                    force_data=parameter.xml_data(), increment=increment
                ),
            )

        # not common so check if we need the section
        if (
            molecule.RBTorsionForce.n_parameters > 0
            or molecule.ImproperRBTorsionForce.n_parameters > 0
        ):
            if molecule.RBTorsionForce.openmm_group() not in force_by_type:
                new_rb_torsion_force = ET.SubElement(
                    root,
                    molecule.RBTorsionForce.openmm_group(),
                    attrib=molecule.RBTorsionForce.xml_data(),
                )
                force_by_type[
                    molecule.RBTorsionForce.openmm_group()
                ] = new_rb_torsion_force

            RBTorsion = force_by_type[molecule.RBTorsionForce.openmm_group()]
            for parameter in molecule.RBTorsionForce:
                ET.SubElement(
                    RBTorsion,
                    parameter.openmm_type(),
                    attrib=_update_increment(
                        force_data=parameter.xml_data(), increment=increment
                    ),
                )
            for parameter in molecule.ImproperRBTorsionForce:
                ET.SubElement(
                    RBTorsion,
                    parameter.openmm_type(),
                    attrib=_update_increment(
                        force_data=parameter.xml_data(), increment=increment
                    ),
                )

        for i, site in enumerate(molecule.extra_sites):
            site_data = site.xml_data()
            site_data["index"] = str(i + molecule.n_atoms)
            ET.SubElement(resiude, site.openmm_type(), attrib=site_data)

        if molecule.NonbondedForce.openmm_group() not in force_by_type:
            new_nb_force = ET.SubElement(
                root,
                molecule.NonbondedForce.openmm_group(),
                attrib=molecule.NonbondedForce.xml_data(),
            )
            force_by_type[molecule.NonbondedForce.openmm_group()] = new_nb_force

        NonbondedForce = force_by_type[molecule.NonbondedForce.openmm_group()]
        for parameter in molecule.NonbondedForce:
            # work out if the atom is being optimised
            rfree_code = _get_parameter_code(
                molecule=molecule, atom_index=parameter.atoms[0]
            )
            atom_data = {
                "charge": str(parameter.charge),
                "sigma": str(parameter.sigma),
                "epsilon": str(parameter.epsilon),
                "type": f"QUBE_{str(parameter.atoms[0] + increment)}",
            }
            if rfree_code in parameters or fit_ab:
                # keep track of present codes to optimise
                rfree_codes.add(rfree_code)
                # this is to be refit
                atom = molecule.atoms[parameter.atoms[0]]
                eval_string = _get_eval_string(
                    atom=atom,
                    rfree_data=rfree_data[rfree_code],
                    a_and_b=fit_ab,
                    alpha_ref=rfree_data["alpha"],
                    beta_ref=rfree_data["beta"],
                    rfree_code=rfree_code if rfree_code in parameters else None,
                )
                atom_data["parameter_eval"] = eval_string
                atom_data["volume"] = str(atom.aim.volume)
                atom_data["bfree"] = str(rfree_data[rfree_code]["b_free"])
                atom_data["vfree"] = str(rfree_data[rfree_code]["v_free"])

            ET.SubElement(NonbondedForce, parameter.openmm_type(), attrib=atom_data)

        for i, site in enumerate(molecule.extra_sites, start=1 + increment):
            site_name = f"v-site{i}"
            ET.SubElement(
                NonbondedForce,
                "Atom",
                attrib={
                    "charge": str(site.charge),
                    "epsilon": "0",
                    "sigma": "1",
                    "type": site_name,
                },
            )
        increment += molecule.n_atoms

    # add the ForceBalance tags
    ForceBalance = ET.SubElement(root, "ForceBalance")
    for parameter_to_fit in parameters:
        if parameter_to_fit != "AB" and parameter_to_fit in rfree_codes:
            fit_data = {
                f"{parameter_to_fit.lower()}free": str(
                    rfree_data[parameter_to_fit]["r_free"]
                ),
                "parameterize": f"{parameter_to_fit.lower()}free",
            }
            ET.SubElement(ForceBalance, f"{parameter_to_fit}Element", attrib=fit_data)
    if fit_ab:
        ET.SubElement(
            ForceBalance, "xalpha", alpha=str(rfree_data["alpha"]), parameterize="alpha"
        )
        ET.SubElement(
            ForceBalance, "xbeta", beta=str(rfree_data["beta"]), parameterize="beta"
        )

    return ET.ElementTree(root)


def _get_eval_string(
    atom: "Atom",
    rfree_data: Dict[str, str],
    a_and_b: bool,
    alpha_ref: str,
    beta_ref: str,
    rfree_code: Optional[str] = None,
) -> str:
    """
    Create the parameter eval string used by ForceBalance to calculate the sigma and epsilon parameters.
    """

    if a_and_b:
        alpha = "PARM['xalpha/alpha']"
        beta = "PARM['xbeta/beta']"
    else:
        alpha, beta = alpha_ref, beta_ref
    if rfree_code is not None:
        rfree = f"PARM['{rfree_code}Element/{rfree_code.lower()}free']"
    else:
        rfree = f"{rfree_data['r_free']}"

    eval_string = (
        f"epsilon=({alpha}*{rfree_data['b_free']}*({atom.aim.volume}/{rfree_data['v_free']})**{beta})/(128*{rfree}**6)*{constants.EPSILON_CONVERSION}, "
        f"sigma=2**(5/6)*({atom.aim.volume}/{rfree_data['v_free']})**(1/3)*{rfree}*{constants.SIGMA_CONVERSION}"
    )

    return eval_string


def _find_molecules_and_rfrees() -> Tuple[
    List["Ligand"], Dict[str, Dict[str, Union[str, float]]]
]:
    """
    Loop over the local directories looking for qubekit WorkFlowResults and extract the ligands and a list of all
    unique free params used to parameterise the molecules.
    """
    molecules = []
    element_params = {}

    for folder in os.listdir("."):
        if os.path.isdir(folder) and folder.startswith("QUBEKit_"):
            workflow_result = WorkFlowResult.parse_file(
                os.path.join(folder, "workflow_result.json")
            )
            molecules.append(workflow_result.current_molecule)
            lj_stage = workflow_result.results["non_bonded"].stage_settings
            element_params.update(lj_stage["free_parameters"])
            element_params["alpha"] = lj_stage["alpha"]
            element_params["beta"] = lj_stage["beta"]

    print(f"{len(molecules)} molecules found, combining...")
    return molecules, element_params


def _update_increment(force_data: Dict[str, str], increment: int) -> Dict[str, str]:
    """
    Update the increment on some force data.
    """
    for key, value in force_data.items():
        if key.startswith("class"):
            force_data[key] = str(int(value) + increment)

    return force_data


def _get_parameter_code(molecule: "Ligand", atom_index: int) -> str:
    """
    Work out the rfree code for an atom in a molecule, mainly used for special cases.
    """
    atom = molecule.atoms[atom_index]
    rfree_code = atom.atomic_symbol
    if rfree_code == "H":
        # check if polar
        if molecule.atoms[atom.bonds[0]].atomic_symbol in ["O", "N", "S"]:
            rfree_code = "X"
    return rfree_code
