import unittest

import numpy as np
import torch

from common import physical_loss,physical_energy,group_metrics
from evaluate import cluster_bootstrap


class PhysicalProtocolTests(unittest.TestCase):
    def test_parameter_gradients_match_for_quadratic_and_linear_regions(self):
        """Affine decoder and target inversion must give the same chain-rule derivative, including Huber's quadratic region."""
        features=torch.tensor([[1.,2.],[2.,-1.],[.5,.25]],dtype=torch.float64)
        weight=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
        raw=features@weight
        sigma=.6274358034133911
        offset=torch.tensor([-5.,-7.,-3.],dtype=torch.float64)
        composition=torch.tensor([2.,1.,.5],dtype=torch.float64)
        target=physical_energy(raw.detach(),sigma,offset,composition)+torch.tensor([.001,.2,-.3],dtype=torch.float64)
        decoded=physical_loss(raw,target,offset,composition,sigma)
        encoded=physical_loss(raw,target,offset,composition,sigma,encoded=True)
        left=torch.autograd.grad(decoded,weight,retain_graph=True)[0]
        right=torch.autograd.grad(encoded,weight)[0]
        self.assertAlmostEqual(float(decoded.detach()),float(encoded.detach()),places=12)
        torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)

    def test_composition_offset_preserves_centered_predictions(self):
        """An identical offset within each composition changes absolute errors only."""
        truth=np.array([0.,.02,.1,.15])
        prediction=np.array([.04,.07,.12,.18])
        groups=np.array(['a','a','b','b'])
        shift=np.array([2.,2.,-1.,-1.])
        left=group_metrics(truth,prediction,groups)
        right=group_metrics(truth,prediction+shift,groups)
        self.assertAlmostEqual(left['ec_mae_mev'],right['ec_mae_mev'])
        self.assertAlmostEqual(left['major_wrong_rate'],right['major_wrong_rate'])

    def test_cluster_bootstrap_keeps_singletons_and_cluster_sizes(self):
        """Group a has [1,3], group b has [9]; combined mean=13/3, repeated-a mean=2, repeated-b mean=9."""
        values=cluster_bootstrap(np.array([1.,3.,9.]),np.array(['a','a','b']),np.array([[0,1],[0,0],[1,1]]))
        np.testing.assert_allclose(values,[13/3,2.,9.])


if __name__=='__main__':
    unittest.main()
