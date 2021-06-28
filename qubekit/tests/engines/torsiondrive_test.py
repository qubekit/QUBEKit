"""
Test the torsiondrive json api interface.
"""
from typing import Any, Dict

import numpy as np
import pytest

from qubekit.engines import TorsionDriver, optimise_grid_point
from qubekit.molecules import Ligand
from qubekit.utils import constants
from qubekit.utils.datastructures import LocalResource, QCOptions, TorsionScan
from qubekit.utils.file_handling import get_data


@pytest.fixture
def ethane_state() -> Dict[str, Any]:
    """
    build an initial state for a ethane scan.
    """
    mol = Ligand.from_file(get_data("ethane.sdf"))
    bond = mol.find_rotatable_bonds()[0]
    dihedral = mol.dihedrals[bond.indices][0]
    tdriver = TorsionDriver(grid_spacing=15)
    # make the scan data
    dihedral_data = TorsionScan(torsion=dihedral, scan_range=(-165, 180))
    qc_spec = QCOptions(program="rdkit", basis=None, method="uff")
    td_state = tdriver._create_initial_state(
        molecule=mol, dihedral_data=dihedral_data, qc_spec=qc_spec
    )
    return td_state


def test_get_initial_state(tmpdir):
    """
    Make sure we can correctly build a starting state using the torsiondrive api.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("ethane.sdf"))
        bond = mol.find_rotatable_bonds()[0]
        dihedral = mol.dihedrals[bond.indices][0]
        tdriver = TorsionDriver()
        # make the scan data
        dihedral_data = TorsionScan(torsion=dihedral, scan_range=(-165, 180))
        td_state = tdriver._create_initial_state(
            molecule=mol, dihedral_data=dihedral_data, qc_spec=QCOptions()
        )
        assert td_state["dihedrals"] == [
            dihedral,
        ]
        assert td_state["elements"] == [atom.atomic_symbol for atom in mol.atoms]
        assert td_state["dihedral_ranges"] == [
            (-165, 180),
        ]
        assert np.allclose(
            (mol.coordinates * constants.ANGS_TO_BOHR), td_state["init_coords"][0]
        )


def test_optimise_grid_point_and_update(tmpdir, ethane_state):
    """
    Try and perform a single grid point optimisation.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("ethane.sdf"))
        tdriver = TorsionDriver(n_workers=1)
        qc_spec = QCOptions(program="rdkit", basis=None, method="uff")
        local_ops = LocalResource(cores=1, memory=1)
        geo_opt = tdriver._build_geometry_optimiser()
        # get the job inputs
        new_jobs = tdriver._get_new_jobs(td_state=ethane_state)
        coords = new_jobs["-60"][0]
        result = optimise_grid_point(
            geometry_optimiser=geo_opt,
            qc_spec=qc_spec,
            local_options=local_ops,
            molecule=mol,
            coordinates=coords,
            dihedral=ethane_state["dihedrals"][0],
            dihedral_angle=-60,
        )
        new_state = tdriver._update_state(
            td_state=ethane_state,
            result_data=[
                result,
            ],
        )
        next_jobs = tdriver._get_new_jobs(td_state=new_state)
        assert "-75" in next_jobs
        assert "-45" in next_jobs


def test_full_tdrive(tmpdir):
    """
    Try and run a full torsiondrive for ethane with a cheap rdkit method.
    """
    with tmpdir.as_cwd():

        ethane = Ligand.from_file(get_data("ethane.sdf"))
        # make the scan data
        bond = ethane.find_rotatable_bonds()[0]
        dihedral = ethane.dihedrals[bond.indices][0]
        dihedral_data = TorsionScan(torsion=dihedral, scan_range=(-165, 180))
        qc_spec = QCOptions(program="rdkit", basis=None, method="uff")
        local_ops = LocalResource(cores=1, memory=1)
        tdriver = TorsionDriver(
            n_workers=1,
            grid_spacing=60,
        )
        _ = tdriver.run_torsiondrive(
            molecule=ethane,
            dihedral_data=dihedral_data,
            qc_spec=qc_spec,
            local_options=local_ops,
        )


def test_get_new_jobs(ethane_state):
    """
    Make sure that for a given initial state we can get the next jobs to be done.
    """
    tdriver = TorsionDriver()
    new_jobs = tdriver._get_new_jobs(td_state=ethane_state)
    assert "-60" in new_jobs
    assert new_jobs["-60"][0] == pytest.approx(
        [
            -1.44942051524959,
            0.015117815022160003,
            -0.030235630044320005,
            1.44942051524959,
            -0.015117815022160003,
            0.030235630044320005,
            -2.2431058039129903,
            -0.18897268777700002,
            1.8708296089923,
            -2.16562700192442,
            1.78768162637042,
            -0.82203119182995,
            -2.1920831782132,
            -1.5325684978714702,
            -1.18863820611733,
            2.1920831782132,
            1.5306787709937002,
            1.18863820611733,
            2.2431058039129903,
            0.18897268777700002,
            -1.8708296089923,
            2.16562700192442,
            -1.78957135324819,
            0.82014146495218,
        ]
    )
