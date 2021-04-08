import re
from typing import Tuple

from openff.toolkit.typing.chemistry import ChemicalEnvironment, SMIRKSParsingError
from pydantic import Field, validator

from qubekit.molecules import Atom, Bond, Ligand
from qubekit.utils.datastructures import SchemaBase


class AvoidedTorsion(SchemaBase):
    """
    A class to detail torsion substructures which should not be scanned.
    """

    smirks: str = Field(..., description="The smirks pattern which is to be scanned.")

    def __hash__(self):
        return hash(self.smirks)

    @validator("smirks")
    def validate_smirks(cls, smirks: str):
        """
        Make sure the smirks is valid by trying to make an environment from it.
        Also make sure either two or 4 atoms are tagged for the dihedral.
        """
        # validate the smarts with the toolkit
        ChemicalEnvironment.validate_smirks(smirks=smirks)
        # look for tags
        tags = re.findall(":[0-9]]", smirks)
        if len(tags) == 2 or len(tags) == 4:
            return smirks
        else:
            raise SMIRKSParsingError(
                "The smirks pattern passed has an incorrect number of tagged atoms, to identify a bond to not rotate"
                "please tag 2 or 4 atoms."
            )


class TargetTorsion(AvoidedTorsion):
    """
    A class to detail allowed smirks patterns to be drive and the scan range.
    """

    scan_range: Tuple[int, int] = Field(
        (-165, 180), description="The range this smirks pattern should be scanned over."
    )


def find_heavy_torsion(molecule: Ligand, bond: Bond) -> Tuple[int, int, int, int]:
    """
    For a given molecule and bond find a torsion suitable to drive by avoiding hydrogens.

    Args:
        molecule: The molecule we wish to torsion drive.
        bond: The bond in the molecule that should be driven.

    Returns:
        A tuple of atom indices corresponding to a suitable torsion.
    """

    terminal_atoms = {}
    atoms = [molecule.atoms[i] for i in bond.indices]

    for atom in atoms:
        for neighbour in atom.bonds:
            if neighbour not in bond.indices:
                if (
                    molecule.atoms[neighbour].atomic_number
                    > terminal_atoms.get(
                        atom.atom_name,
                        Atom(
                            atomic_number=0,
                            atom_index=0,
                            formal_charge=0,
                            aromatic=False,
                        ),
                    ).atomic_number
                ):
                    terminal_atoms[atom.atom_name] = molecule.atoms[neighbour]
    # now build the torsion
    torsion = [atom.atom_index for atom in terminal_atoms.values()]
    for i, atom in enumerate(atoms, 1):
        torsion.insert(i, atom.atom_index)

    return tuple(torsion)
