import unittest
import numpy as np

from analysis import group_values,metrics


class AnalysisTests(unittest.TestCase):
    def test_mse_decomposition_known_values(self):
        """Residuals [1,3] have mean-bias MSE 4 and within MSE 1, total 5; MAE remains 2."""
        table=group_values(np.zeros(2),np.zeros(2),np.array([1.,3.]),np.array(['g','g']))
        _,decomposition=metrics(table.sum(axis=0),1)
        self.assertAlmostEqual(decomposition['mse_total_ev2_atom2'],5.)
        self.assertAlmostEqual(decomposition['mse_group_bias_ev2_atom2'],4.)
        self.assertAlmostEqual(decomposition['mse_within_ev2_atom2'],1.)
        self.assertAlmostEqual(decomposition['mae_mev_atom'],2000.)

    def test_negligible_target_cannot_define_ratio(self):
        table=group_values(np.array([0.,1e-10]),np.zeros(2),np.array([0.,.1]),np.array(['g','g']))
        correction,_=metrics(table.sum(axis=0),1)
        self.assertTrue(np.isnan(correction['norm_ratio']))
        self.assertTrue(np.isnan(correction['direction_cosine']))

    def test_zero_correction_is_not_a_direction(self):
        table=group_values(np.array([0.,.02]),np.zeros(2),np.zeros(2),np.array(['g','g']))
        correction,_=metrics(table.sum(axis=0),1)
        self.assertEqual(correction['norm_ratio'],0.)
        self.assertTrue(np.isnan(correction['direction_cosine']))

    def test_center_after_cleaning(self):
        """Only retained y=[0,.02] participate; group-centered baseline error is [.01,-.01]."""
        table=group_values(np.array([0.,.02]),np.zeros(2),np.zeros(2),np.array(['g','g']))
        correction,_=metrics(table.sum(axis=0),1)
        self.assertAlmostEqual(correction['ec_mae_mev_atom'],10.)

    def test_exact_target_correction(self):
        table=group_values(np.array([0.,.02]),np.zeros(2),np.array([4.,4.02]),np.array(['g','g']))
        correction,_=metrics(table.sum(axis=0),1)
        self.assertAlmostEqual(correction['norm_ratio'],1.)
        self.assertAlmostEqual(correction['direction_cosine'],1.)
        self.assertAlmostEqual(correction['correction_error_norm_ev_atom'],0.)


if __name__=='__main__':
    unittest.main()
