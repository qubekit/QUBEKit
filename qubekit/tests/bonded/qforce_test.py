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
        assert QForceHessianFitting.is_available() is False


def test_generate_qforce_settings():
    """
    Generate the qforce settings string.
    """
    qf = QForceHessianFitting(combination_rule="amber", charge_scaling=1)
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
        urey = false
        """
    assert settings.read() == StringIO(expected).read()


def test_generate_atom_types(acetone, openff):
    """
    Make sure atom types can be correctly grouped.
    """
    openff.run(acetone)
    atom_types = QForceHessianFitting._generate_atom_types(molecule=acetone)
    assert atom_types == {
        "lj_types": ["C1", "C2", "O3", "C1", "H0", "H0", "H0", "H0", "H0", "H0"],
        "atom_types": {
            "C1": (0.3399669508423535, 0.4577296),
            "C2": (0.3399669508423535, 0.359824),
            "O3": (0.2959921901149463, 0.87864),
            "H0": (0.2649532787749369, 0.06568879999999999),
        },
    }


def test_coumarin_run(tmpdir, coumarin):
    """
    Run coumarin through qforce and make sure we have the expected number of bond, angle, periodic and RB proper
    and improper torsions.
    """
    if not QForceHessianFitting.is_available():
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
