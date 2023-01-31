import numpy as np
import pytest

from qubekit.charges import DDECCharges, ExtractChargeData
from qubekit.molecules import Ligand
from qubekit.nonbonded import LennardJones612, get_protocol
from qubekit.nonbonded.protocols import (
    b_base,
    br_base,
    c_base,
    cl_base,
    f_base,
    h_base,
    i_base,
    n_base,
    o_base,
    p_base,
    s_base,
    si_base,
)
from qubekit.parametrisation import OpenFF
from qubekit.utils import constants
from qubekit.utils.exceptions import MissingRfreeError
from qubekit.utils.file_handling import get_data


def test_lennard_jones612_polar_h_no_ab(methanol):
    """
    Make sure polar h have there parameters correctly calculated when there are no alpha and beta terms.
    """
    lj_protocol = get_protocol(protocol_name="0")
    lj_protocol.run(molecule=methanol)
    # check the polar hydrogen
    assert methanol.NonbondedForce[(5,)].epsilon == 1.814472172136794
    assert methanol.NonbondedForce[(5,)].sigma == 0.12906327125421754
    # check the oxygen
    assert methanol.NonbondedForce[(1,)].epsilon == 0.6193305402659339
    assert methanol.NonbondedForce[(1,)].sigma == 0.2811820692687191


def test_lennard_jones612_polar_h_ab(methanol):
    """
    Make sure polar h have there parameters correctly calculated when there are alpha and beta terms.
    """

    lj_protocol = get_protocol(protocol_name="5b")
    lj_protocol.run(molecule=methanol)
    # check the polar hydrogen
    assert methanol.NonbondedForce[(5,)].epsilon == 0.3621678392735354
    assert methanol.NonbondedForce[(5,)].sigma == 0.15873709816308199
    # check the oxygen
    assert methanol.NonbondedForce[(1,)].epsilon == 0.5341909748395333
    assert methanol.NonbondedForce[(1,)].sigma == 0.2997524661050121


def test_lennard_jones612_no_polar_h_ab(methanol):
    """
    Test that polar H C6 is correctly transferred to the parent atom when requested, and that the alpha and beta
    scaling are correctly applied.
    """

    # use the no polar H protocol with alpha=1 and beta=0
    lj_protocol = get_protocol(protocol_name="4a")
    lj_protocol.run(molecule=methanol)

    # check the polar hydrogen
    assert methanol.NonbondedForce[(5,)].epsilon == 0.0
    assert methanol.NonbondedForce[(5,)].sigma == 1.0
    # check the oxygen
    assert methanol.NonbondedForce[(1,)].epsilon == 0.6883098722469343
    assert methanol.NonbondedForce[(1,)].sigma == 0.29074863733590034
    # calculate what epsilon should be for oxygen with no alpha and beta
    h_c6 = (
        lj_protocol.free_parameters["H"].b_free
        * (methanol.atoms[5].aim.volume / lj_protocol.free_parameters["H"].v_free) ** 2
    )
    o_c6 = (
        lj_protocol.free_parameters["O"].b_free
        * (methanol.atoms[1].aim.volume / lj_protocol.free_parameters["O"].v_free) ** 2
    )
    o_c6_prime = (np.sqrt(o_c6) + np.sqrt(h_c6)) ** 2
    o_epsilon = (
        o_c6_prime
        / (
            128
            * (
                (methanol.atoms[1].aim.volume / lj_protocol.free_parameters["O"].v_free)
                ** 2
            )
            * lj_protocol.free_parameters["O"].r_free ** 6
        )
        * constants.EPSILON_CONVERSION
    )
    assert pytest.approx(o_epsilon, abs=1e-7) == methanol.NonbondedForce[(1,)].epsilon

    # now change alpha and beta and check that epsilon has changed but sigma is the same
    lj_protocol.alpha = 1.1
    lj_protocol.beta = 0.4
    lj_protocol.run(molecule=methanol)
    # check the polar hydrogen
    assert methanol.NonbondedForce[(5,)].epsilon == 0.0
    assert methanol.NonbondedForce[(5,)].sigma == 1.0
    # check the oxygen
    assert methanol.NonbondedForce[(1,)].epsilon == 0.7517401841871586
    assert methanol.NonbondedForce[(1,)].sigma == 0.29074863733590034

    # calculate what epsilon should be with alpha and beta
    # scale H_C6
    h_c6 *= (
        lj_protocol.alpha
        * (methanol.atoms[5].aim.volume / lj_protocol.free_parameters["H"].v_free)
        ** lj_protocol.beta
    )
    # scale O_C6
    o_c6 *= (
        lj_protocol.alpha
        * (methanol.atoms[1].aim.volume / lj_protocol.free_parameters["O"].v_free)
        ** lj_protocol.beta
    )
    o_c6_prime = (np.sqrt(o_c6) + np.sqrt(h_c6)) ** 2
    o_epsilon = (
        o_c6_prime
        / (
            128
            * (
                (methanol.atoms[1].aim.volume / lj_protocol.free_parameters["O"].v_free)
                ** 2
            )
            * lj_protocol.free_parameters["O"].r_free ** 6
        )
        * constants.EPSILON_CONVERSION
    )
    assert pytest.approx(o_epsilon, abs=1e-7) == methanol.NonbondedForce[(1,)].epsilon


def test_lennard_jones612(tmpdir):
    """
    Make sure that we can reproduce some reference values using the LJ612 class
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("chloromethane.pdb"))
        # get some initial Nonbonded values
        OpenFF().run(molecule=mol)
        # get some aim reference data
        ExtractChargeData.extract_charge_data_chargemol(
            molecule=mol, dir_path=get_data(""), ddec_version=6
        )
        # apply symmetry to the reference data
        DDECCharges.apply_symmetrisation(molecule=mol)
        # calculate the new LJ terms
        LennardJones612(
            lj_on_polar_h=False,
            # qubekit 1 legacy parameters
            free_parameters={
                "H": h_base(r_free=1.64),
                "C": c_base(r_free=2.08),
                "Cl": cl_base(r_free=1.88),
            },
        ).run(molecule=mol)
        # make sure we get out expected reference values
        assert mol.NonbondedForce[(0,)].sigma == 0.3552211069814666
        assert mol.NonbondedForce[(0,)].epsilon == 0.25918723101839924
        assert mol.NonbondedForce[(1,)].sigma == 0.33888067968663566
        assert mol.NonbondedForce[(1,)].epsilon == 0.9650542683335082
        assert mol.NonbondedForce[(2,)].sigma == 0.22192905304751342
        assert mol.NonbondedForce[(2,)].epsilon == 0.15047278650152818


def test_lennard_jones_missing_polar(acetone):
    """
    Make sure an error is raised if we request lj terms on hydrogen but do not supply an rfree.
    """

    lj = LennardJones612(
        lj_on_polar_h=True,
        free_parameters={
            "H": h_base(r_free=1),
            "C": c_base(r_free=2.00),
            "O": o_base(r_free=1.65),
        },
    )

    with pytest.raises(MissingRfreeError, match="Please supply Rfree data for polar"):
        lj.check_element_coverage(molecule=acetone)


def test_lennard_jones_missing_element(acetone):
    """
    Make sure an error is raised if we try and derive parameters for an element we have no reference for.
    """

    lj = LennardJones612(
        lj_on_polar_h=False,
        # missing oxygen
        free_parameters={"C": c_base(r_free=2.00), "H": h_base(r_free=1)},
    )

    with pytest.raises(
        MissingRfreeError, match="The following elements have no reference Rfree"
    ):
        lj.check_element_coverage(molecule=acetone)


def test_get_protocol():
    """
    Make sure we get the correct protocol when requested and that the defaults are correct.
    """

    model0 = get_protocol(protocol_name="0")

    assert model0.lj_on_polar_h is True
    assert model0.free_parameters["X"].r_free == 1.083
    assert model0.free_parameters["H"].r_free == 1.738
    assert model0.free_parameters["C"].r_free == 2.008
    assert model0.free_parameters["N"].r_free == 1.765
    assert model0.free_parameters["O"].r_free == 1.499
    assert model0.alpha == 1
    assert model0.beta == 0


@pytest.mark.parametrize(
    "base_func,expected",
    [
        pytest.param(h_base, (7.6, 6.5)),
        pytest.param(b_base, (46.7, 99.5)),
        pytest.param(c_base, (34.4, 46.6)),
        pytest.param(n_base, (25.9, 24.2)),
        pytest.param(o_base, (22.1, 15.6)),
        pytest.param(f_base, (18.2, 9.5)),
        pytest.param(p_base, (84.6, 185)),
        pytest.param(s_base, (75.2, 134.0)),
        pytest.param(cl_base, (65.1, 94.6)),
        pytest.param(br_base, (95.7, 162.0)),
        pytest.param(si_base, (101.64, 305)),
        pytest.param(i_base, (153.8, 385.0)),
    ],
)
def test_element_base(base_func, expected):
    """
    Make sure each element base gives the correct values.
    """
    rfree = np.random.random()
    element_data = base_func(rfree)
    assert (element_data.v_free, element_data.b_free, element_data.r_free) == (
        *expected,
        rfree,
    )
