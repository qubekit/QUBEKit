from qubekit.charges import DDECCharges, ExtractChargeData
from qubekit.molecules import Ligand
from qubekit.nonbonded import LennardJones612
from qubekit.parametrisation import OpenFF
from qubekit.utils.file_handling import get_data


def test_lennard_jones612(tmpdir):
    """
    Make sure that we can reproduce some reference values using the LJ612 class
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("chloromethane.pdb"))
        # get some initial Nonbonded values
        OpenFF().run(molecule=mol)
        # get some aim reference data
        ExtractChargeData.read_files(
            molecule=mol, dir_path=get_data(""), charges_engine="chargemol"
        )
        # apply symmetry to the reference data
        DDECCharges.apply_symmetrisation(molecule=mol)
        # calculate the new LJ terms
        LennardJones612().run(molecule=mol)
        # make sure we get out expected reference values
        assert mol.NonbondedForce[(0,)].sigma == 0.3552211069814666
        assert mol.NonbondedForce[(0,)].epsilon == 0.25918723101839924
        assert mol.NonbondedForce[(1,)].sigma == 0.33888067968663566
        assert mol.NonbondedForce[(1,)].epsilon == 0.9650542683335082
        assert mol.NonbondedForce[(2,)].sigma == 0.22192905304751342
        assert mol.NonbondedForce[(2,)].epsilon == 0.15047278650152818
