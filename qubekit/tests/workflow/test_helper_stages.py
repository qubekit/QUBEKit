from qubekit.utils.datastructures import LocalResource, QCOptions
from qubekit.workflow.helper_stages import Hessian


def test_ani_hessian(acetone, tmpdir):
    """
    Test computing the hessian using ml, note this will not provide the wbo matrix
    """
    with tmpdir.as_cwd():
        # make sure the hessian has not already been assigned by mistake
        assert acetone.hessian is None
        spec = QCOptions(program="torchani", basis=None, method="ani2x")
        options = LocalResource(cores=1, memory=1)
        hes_stage = Hessian()
        hes_stage.run(molecule=acetone, qc_spec=spec, local_options=options)
        assert acetone.wbo is None
        assert acetone.hessian is not None
        assert acetone.hessian.shape == (30, 30)
