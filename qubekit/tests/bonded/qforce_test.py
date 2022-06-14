from io import StringIO

import pytest

from qubekit.bonded import QForceHessianFitting


def test_is_available():
    """
    Make sure we can correctly detect when qforce is installed.
    """
    try:
        import qforce

        assert QForceHessianFitting.is_available() is True
    except (ModuleNotFoundError, ImportError):
        pass


def test_generate_qforce_settings():
    """
    Generate the qforce settings string.
    """
    qf = QForceHessianFitting(
        combination_rule="amber", charge_scaling=1, use_urey_bradley=False
    )
    settings = qf._generate_settings()
    expected = """
        [ff]
        lennard_jones = ext
        ext_lj_fudge = 0.5
        ext_q_fudge = 0.8333
        ext_comb_rule = 2
        ext_h_cap = H0
        charge_scaling = 1.0
        [terms]
        urey = False
        """
    assert settings.read() == StringIO(expected).read()


def test_generate_atom_types(acetone, openff, tmpdir):
    """
    Make sure atom types can be correctly grouped.
    """
    with tmpdir.as_cwd():
        openff.run(acetone)
        atom_types = QForceHessianFitting._generate_atom_types(molecule=acetone)
        assert atom_types == {
            "lj_types": ["C1", "C2", "O3", "C1", "H0", "H0", "H0", "H0", "H0", "H0"],
            "atom_types": {
                "C1": (0.3379531761626621, 0.45538911611061844),
                "C2": (0.3480646886945065, 0.3635030558377792),
                "O3": (0.30398122050658094, 0.8795023257036865),
                "H0": (0.2644543413268125, 0.06602135607582665),
            },
        }


def test_coumarin_run(tmpdir, coumarin):
    """
    Run coumarin through qforce and make sure we have the expected number of bond, angle, periodic and RB proper
    and improper torsions.
    """
    try:
        QForceHessianFitting.is_available()
    except ModuleNotFoundError:
        pytest.skip("QForce is not available skipping test.")

    with tmpdir.as_cwd():
        qf = QForceHessianFitting()
        qf.run(coumarin)
        assert coumarin.BondForce.n_parameters == coumarin.n_bonds
        assert coumarin.AngleForce.n_parameters == coumarin.n_angles
        assert coumarin.TorsionForce.n_parameters == 45
        assert coumarin.RBTorsionForce.n_parameters == 6


def test_messages():
    """
    Make sure the start and end massages can be displayed.
    """
    qf = QForceHessianFitting()
    assert "Starting internal hessian fitting using QForce." == qf.start_message()
    assert (
        "Internal hessian fitting finished, saving parameters." == qf.finish_message()
    )
