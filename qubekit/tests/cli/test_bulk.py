from qubekit.cli.bulk import create
from qubekit.molecules import Ligand
from qubekit.utils.file_handling import get_data


def test_bulk_create(run_cli, tmpdir):
    """
    Test creating a csv file for a bulk run.
    """
    molecules = ["acetone.sdf", "bace0.sdf", "benzene.sdf"]
    for mol_file in molecules:
        mol = Ligand.from_file(get_data(mol_file))
        mol.to_file(mol_file)

    output = run_cli.invoke(create, args=["test.csv"])
    assert output.exit_code == 0
