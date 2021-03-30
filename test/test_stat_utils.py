import utils.stat_utils as su

import unittest
import numpy as np


class StatUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_gaussian_estimator(self):
        ge = su.GaussianEstimator()

        ge.update(10.0, 1.0)
        self.assertListEqual([ge.get_count(), ge.get_mean(), ge.get_var(), ge.get_std()], [1.0, 10.0, 0.0, 0.0])

        ge.update(10.0, 1.0)
        self.assertListEqual([ge.get_count(), ge.get_mean(), ge.get_var(), ge.get_std()], [2.0, 10.0, 0.0, 0.0])

        ge.update(5.0, 1.0)
        all_vals = [10.0, 10.0, 5.0]
        np.testing.assert_almost_equal([ge.get_count(), ge.get_mean(), ge.get_var(), ge.get_std()], [3.0, np.mean(all_vals), np.var(all_vals) * 3.0, np.std(all_vals)], decimal=3)

        ge.update(3.0, 10.0)
        all_vals += [3.0] * 10
        np.testing.assert_almost_equal([ge.get_count(), ge.get_mean(), ge.get_var(), ge.get_std()], [13.0, np.mean(all_vals), np.var(all_vals) * 13.0, np.std(all_vals)], decimal=3)

        ge.update(-1.0, 10.0)
        all_vals += [-1.0] * 10
        np.testing.assert_almost_equal([ge.get_count(), ge.get_mean(), ge.get_var(), ge.get_std()], [23.0, np.mean(all_vals), np.var(all_vals) * 23.0, np.std(all_vals)], decimal=3)

    def test_gaussian_stats(self):
        atts, num_cls = np.arange(3), 2
        gs = su.Statistics(atts, num_cls, su.GaussianEstimator)
        self.assertEqual((len(gs.att_stats), len(gs.att_stats[0])), (len(atts), num_cls))
        self.assertEqual(gs.att_extr_stats.shape, (len(atts), 2))
        self.assertEqual(gs.get_cls_count(), 0.0)
        self.assertEqual(len(gs.cls_counts), 2)
        self.assertEqual(gs.cls_counts[0], 0.0)
        self.assertEqual(gs.cls_counts[1], 0.0)

        gs.update(np.array([3.0, 2.0, 1.0]), 0, 1.0)
        self.assertEqual(gs.get_cls_count(), 1.0)
        self.assertEqual(len(gs.cls_counts), 2)
        self.assertEqual(gs.cls_counts[0], 1.0)
        self.assertEqual(gs.cls_counts[1], 0.0)

        self.assertEqual((len(gs.att_stats), len(gs.att_stats[0])), (len(atts), num_cls))
        est = gs.get_estimator(0, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 3.0, 0.0, 0.0])
        est = gs.get_estimator(0, 1)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [0.0, 0.0, 0.0, 0.0])
        self.assertListEqual(gs.att_extr_stats[0].tolist(), [3.0, 3.0])

        est = gs.get_estimator(1, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 2.0, 0.0, 0.0])
        est = gs.get_estimator(2, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 1.0, 0.0, 0.0])

        gs.update(np.array([1.0, 2.0, 3.0]), 1, 1.0)
        self.assertEqual(gs.get_cls_count(), 2.0)
        self.assertEqual(len(gs.cls_counts), 2)
        self.assertEqual(gs.cls_counts[0], 1.0)
        self.assertEqual(gs.cls_counts[1], 1.0)

        self.assertEqual((len(gs.att_stats), len(gs.att_stats[0])), (len(atts), num_cls))
        est = gs.get_estimator(0, 1)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 1.0, 0.0, 0.0])
        est = gs.get_estimator(0, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 3.0, 0.0, 0.0])
        self.assertListEqual(gs.att_extr_stats[0].tolist(), [1.0, 3.0])

        est = gs.get_estimator(1, 1)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 2.0, 0.0, 0.0])
        est = gs.get_estimator(2, 1)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 3.0, 0.0, 0.0])

        ge = su.GaussianEstimator(100, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(ge.get_cdf(-0.9), 0.18406, delta=0.00001)
        self.assertAlmostEqual(ge.get_cdf(-0.6), 0.27425, delta=0.00001)
        self.assertAlmostEqual(ge.get_cdf(-0.3), 0.38209, delta=0.00001)
        self.assertEqual(ge.get_cdf(0.0), 0.5)
        self.assertAlmostEqual(ge.get_cdf(0.3), 0.61791, delta=0.00001)
        self.assertAlmostEqual(ge.get_cdf(0.6), 0.72575, delta=0.00001)
        self.assertAlmostEqual(ge.get_cdf(0.9), 0.81594, delta=0.00001)

        ge = su.GaussianEstimator(100, mean=1.0, std=0.5)
        self.assertAlmostEqual(ge.get_pdf(0.25), 0.25904, delta=0.0001)
        ge = su.GaussianEstimator(100, mean=1.0, std=0.5)
        self.assertAlmostEqual(ge.get_pdf(1.0), 0.79788, delta=0.0001)
        self.assertAlmostEqual(ge.get_pdf(1.5), 0.48394, delta=0.0001)
        ge = su.GaussianEstimator(100, mean=5.0, std=0.0)
        self.assertEqual(ge.get_pdf(5.0), 1.0)
        self.assertEqual(ge.get_pdf(4.0), 0.0)

    def test_incremental_gaussian_stats(self):
        gs = su.Statistics(np.arange(3), 2, su.GaussianEstimator)

        gs.update(np.array([3.0, 2.0, 1.0]), 3, 1.0)
        self.assertEqual(gs.get_cls_count(), 1.0)
        self.assertEqual(len(gs.cls_counts), 4)
        self.assertEqual(gs.cls_counts[0], 0.0)
        self.assertEqual(gs.cls_counts[1], 0.0)
        self.assertEqual(gs.cls_counts[2], 0.0)
        self.assertEqual(gs.cls_counts[3], 1.0)

        self.assertEqual((len(gs.att_stats), len(gs.att_stats[0])), (3, 4))
        est = gs.get_estimator(0, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [0.0, 0.0, 0.0, 0.0])
        est = gs.get_estimator(0, 1)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [0.0, 0.0, 0.0, 0.0])
        est = gs.get_estimator(0, 2)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [0.0, 0.0, 0.0, 0.0])
        est = gs.get_estimator(0, 3)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 3.0, 0.0, 0.0])
        self.assertListEqual(gs.att_extr_stats[0].tolist(), [3.0, 3.0])

        est = gs.get_estimator(1, 3)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 2.0, 0.0, 0.0])
        est = gs.get_estimator(2, 3)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 1.0, 0.0, 0.0])

    def test_rnd_att_gaussian_stats(self):
        atts = np.array([1, 4, 5, 9])
        gs = su.Statistics(atts, 2, su.GaussianEstimator)

        gs.update(np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), 0, 1.0)
        est = gs.get_estimator(1, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 8.0, 0.0, 0.0])
        est = gs.get_estimator(4, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 5.0, 0.0, 0.0])
        est = gs.get_estimator(5, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 4.0, 0.0, 0.0])
        est = gs.get_estimator(9, 0)
        self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [1.0, 0.0, 0.0, 0.0])

        for att_idx in np.array([1, 4, 5, 9]):
            est = gs.get_estimator(att_idx, 1)
            self.assertListEqual([est.get_count(), est.get_mean(), est.get_var(), est.get_std()], [0.0, 0.0, 0.0, 0.0])

    def test_calculate_split_entropy(self):
        atts, num_cls = np.arange(3), 2
        gs = su.Statistics(atts, num_cls, su.GaussianEstimator)

        c0_data = np.random.multivariate_normal(np.array([4.0, -10.0, 5.0]), np.eye(3) * 1.0, 100)
        c1_data = np.random.multivariate_normal(np.array([6.0, 10.0, 5.0]), np.eye(3) * 1.0, 100)
        for x in c0_data: gs.update(x, 0, 1.0)
        for x in c1_data: gs.update(x, 1, 1.0)

        e1 = gs.calc_split_weighted_entropy(gs, 1, -10.0, num_cls)[0]
        e2 = gs.calc_split_weighted_entropy(gs, 1, 0.0, num_cls)[0]
        e3 = gs.calc_split_weighted_entropy(gs, 1, 10.0, num_cls)[0]
        e4 = gs.calc_split_weighted_entropy(gs, 1, 30.0, num_cls)[0]

        self.assertLess(e2, e1)
        self.assertLess(e2, e3)
        self.assertLess(e2, e4)
        self.assertLess(e3, e4)

        # import torch
        # from torch import Tensor
        # print(torch.distributions.Categorical(Tensor([0.5, 0.5])).entropy())
        # print(torch.distributions.Categorical(Tensor([0.25, 0.75])).entropy())
        # print(torch.distributions.Categorical(Tensor([0.0, 1.0])).entropy())
        # print(torch.distributions.Categorical(Tensor([0.001, 0.002])).entropy())
        # print(torch.distributions.Categorical(Tensor([0.00001, 0.002])).entropy())

    def test_split_statistics(self):
        stat_atts, num_cls = np.arange(3), 2
        split_att_idx, split_att_val = 1, 0.0
        st = su.Statistics(stat_atts, num_cls, su.GaussianEstimator, att_split_est=False)
        c0_data = np.random.multivariate_normal(np.array([4.0, -10.0, 5.0]), np.eye(3) * 1.0, 100)
        c1_data = np.random.multivariate_normal(np.array([6.0, 10.0, 5.0]), np.eye(3) * 1.0, 100)
        for x in c0_data: st.update(x, 0, 1.0)
        for x in c1_data: st.update(x, 1, 1.0)

        st_left, st_right = st.split(split_att_idx, split_att_val, l_prob=np.array([0.45, 0.05]), r_prob=np.array([0.05, 0.45]))

        self.assertIsNotNone(st_left)
        self.assertEqual(st_left.get_cls_count(0), 90.0)
        self.assertEqual(st_left.get_cls_count(1), 10.0)
        for att_idx in stat_atts:
            self.assertEqual(st_left.get_estimator(att_idx, 0).get_count(), 0.0)
            self.assertEqual(st_left.get_estimator(att_idx, 1).get_count(), 0.0)
            self.assertListEqual(st_left.get_att_extr(att_idx).tolist(), [float('inf'), float('-inf')])

        self.assertIsNotNone(st_right)
        self.assertEqual(st_right.get_cls_count(0), 10.0)
        self.assertEqual(st_right.get_cls_count(1), 90.0)
        for att_idx in stat_atts:
            self.assertEqual(st_right.get_estimator(att_idx, 0).get_count(), 0.0)
            self.assertEqual(st_right.get_estimator(att_idx, 1).get_count(), 0.0)
            self.assertListEqual(st_right.get_att_extr(att_idx).tolist(), [float('inf'), float('-inf')])

        st = su.Statistics(stat_atts, num_cls, su.GaussianEstimator, att_split_est=True)
        for x in c0_data: st.update(x, 0, 1.0)
        for x in c1_data: st.update(x, 1, 1.0)

        st_left, st_right = st.split(split_att_idx, split_att_val, l_prob=np.array([0.45, 0.05]), r_prob=np.array([0.05, 0.45]))

        self.assertIsNotNone(st_left)
        self.assertEqual(st_left.get_cls_count(0), 90.0)
        self.assertEqual(st_left.get_cls_count(1), 10.0)
        for att_idx in stat_atts:
            self.assertEqual(st_left.get_estimator(att_idx, 0).get_count(), 90.0)
            self.assertEqual(st_left.get_estimator(att_idx, 1).get_count(), 10.0)

            if att_idx != split_att_idx:
                self.assertListEqual(st_left.get_att_extr(att_idx).tolist(), st.get_att_extr(att_idx).tolist())
            else:
                self.assertListEqual(st_left.get_att_extr(att_idx).tolist(), [st.get_att_extr(att_idx).tolist()[0], split_att_val])

        self.assertIsNotNone(st_right)
        self.assertEqual(st_right.get_cls_count(0), 10.0)
        self.assertEqual(st_right.get_cls_count(1), 90.0)
        for att_idx in stat_atts:
            self.assertEqual(st_right.get_estimator(att_idx, 0).get_count(), 10.0)
            self.assertEqual(st_right.get_estimator(att_idx, 1).get_count(), 90.0)

            if att_idx != split_att_idx:
                self.assertListEqual(st_right.get_att_extr(att_idx).tolist(), st.get_att_extr(att_idx).tolist())
            else:
                self.assertListEqual(st_right.get_att_extr(att_idx).tolist(), [split_att_val, st.get_att_extr(att_idx).tolist()[1]])