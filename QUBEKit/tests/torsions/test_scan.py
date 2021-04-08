"""
Torsion Scan set up and run tests.
"""

import numpy as np
import pytest
from openff.toolkit.typing.chemistry import SMIRKSParsingError

from QUBEKit.engines import TorsionDriver
from QUBEKit.molecules import Ligand
from QUBEKit.torsions import TorsionScan1D, find_heavy_torsion
from QUBEKit.utils.file_handling import get_data
from QUBEKit.utils.helpers import check_proper_torsion


def test_torsion_finder_single():
    """
    Make sure we can find a non-hydrogen torsion in a molecule with a rotatble bond.
    """
    mol = Ligand.from_file(get_data("biphenyl.sdf"))
    # there should only be one bond here
    bond = mol.find_rotatable_bonds()[0]
    torsion = find_heavy_torsion(molecule=mol, bond=bond)
    # make sure no atoms are hydrogen
    for t in torsion:
        assert mol.atoms[t].atomic_number != 1
    check_proper_torsion(torsion=torsion, molecule=mol)


def test_torsion_finder_multiple():
    """
    Find non hydrogen torsions for multiple rotatable bonds.
    """
    mol = Ligand.from_smiles("CCO", "ethanol")
    bonds = mol.find_rotatable_bonds()
    for bond in bonds:
        torsion = find_heavy_torsion(molecule=mol, bond=bond)
        check_proper_torsion(torsion=torsion, molecule=mol)


def test_is_available():
    """
    The TorsionScan should always be available as it uses default packages.
    """
    assert TorsionScan1D.is_available() is True


def test_adding_torsions():
    """Make sure we can correctly add new torsion targets."""

    t_scan = TorsionScan1D()
    t_scan.clear_special_torsions()
    # add a ring range limited scan
    t_scan.add_special_torsion(smirks="[*:1]:[*:2]", scan_range=(-40, 40))


def test_adding_torsions_bad():
    """Make sure an error is raised when adding a bad torsion."""

    t_scan = TorsionScan1D()
    with pytest.raises(SMIRKSParsingError):
        t_scan.add_special_torsion(smirks="[C]-[C]", scan_range=(0, 180))


def test_adding_avoided_torsion():
    """make sure we can add new torsions to avoid."""
    t_scan = TorsionScan1D()
    t_scan.clear_avoided_torsions()
    # avoid all double bonds
    t_scan.add_avoided_torsion(smirks="[*:1]=[*:2]")


def test_torsion_special_case_double():
    """
    Make sure special cases changes the scan range for a bond.
    """
    mol = Ligand.from_smiles("CO", "methanol")
    t_scan = TorsionScan1D()
    t_scan.clear_avoided_torsions()
    # add the special case with non-default range
    t_scan.add_special_torsion(smirks="[*:1]-[OH1:2]", scan_range=(0, 180))
    # get the one rotatable bond
    bond = mol.find_rotatable_bonds()[0]
    scan_range = t_scan._get_scan_range(molecule=mol, bond=bond)
    assert scan_range == (0, 180)


def test_torsion_special_case_quad():
    """
    Make sure the special case changes the scan range when a full torsion is described.
    """
    mol = Ligand.from_file(get_data("biphenyl.sdf"))
    t_scan = TorsionScan1D()
    t_scan.clear_avoided_torsions()
    # add the special case with non-default range
    # this will target the bridgehead bond
    t_scan.add_special_torsion(smirks="[c:1]:[c:2]-[c:3]:[c:4]", scan_range=(0, 180))
    # get the one rotatable bond
    bond = mol.find_rotatable_bonds()[0]
    scan_range = t_scan._get_scan_range(molecule=mol, bond=bond)
    assert scan_range == (0, 180)


def test_no_dihedrals(tmpdir):
    """
    Make sure no torsion drives are run when they are all filtered.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("ethane.sdf"))
        t_scan = TorsionScan1D()
        result_mol = t_scan.run(molecule=mol)
        assert result_mol.qm_scans is None
        assert np.allclose(result_mol.coordinates, mol.coordinates)


def test_single_dihedral(tmpdir):
    """Test running a torsiondrive for a molecule with one bond."""
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("ethane.sdf"))
        # build a scanner with grid spacing 60 and clear out avoided methyl
        tdrive = TorsionDriver(
            program="rdkit",
            method="uff",
            basis=None,
            cores=1,
            memory=1,
            n_workers=1,
            grid_spacing=60,
        )
        t_scan = TorsionScan1D(torsion_driver=tdrive)
        t_scan.clear_avoided_torsions()
        result_mol = t_scan.run(molecule=mol)
        assert len(result_mol.qm_scans) == 1
        # make sure input molecule coords were not changed
        assert np.allclose(mol.coordinates, result_mol.coordinates)


def test_double_dihedral(tmpdir):
    """Test running a molecule with two rotatable bonds."""
    with tmpdir.as_cwd():
        mol = Ligand.from_smiles("CCO", "ethanol")
        # build a scanner with grid spacing 60 and clear out avoided methyl
        tdrive = TorsionDriver(
            program="rdkit",
            method="uff",
            basis=None,
            cores=1,
            memory=1,
            n_workers=1,
            grid_spacing=60,
        )
        t_scan = TorsionScan1D(torsion_driver=tdrive)
        t_scan.clear_avoided_torsions()
        result_mol = t_scan.run(molecule=mol)
        assert len(result_mol.qm_scans) == 2
        # make sure input molecule coords were not changed
        assert np.allclose(mol.coordinates, result_mol.coordinates)
