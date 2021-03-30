from utils.cls_utils import ClassificationUtils as cu
from utils.stat_utils import Statistics, GaussianEstimator
import numpy as np
import unittest


class ClassificationUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_majority_class_prob(self):
        prob = cu.majority_class_prob(np.array([65.0, 15.0, 20.0]), 100.0)
        np.testing.assert_almost_equal(prob.tolist(), [0.65, 0.15, 0.2], decimal=2)

        prob = cu.majority_class_prob(np.array([0.0, 0.0, 0.0]), 0.0)
        np.testing.assert_almost_equal(prob.tolist(), [0.0, 0.0, 0.0], decimal=2)

    def test_naive_bayes_prob(self):
        stats = Statistics(np.arange(2), 2, GaussianEstimator)
        c0_data = np.random.multivariate_normal(np.array([10.0, -10.0]), np.eye(2) * 100.0, 100)
        c1_data = np.random.multivariate_normal(np.array([-10.0, 10.0]), np.eye(2) * 100.0, 100)
        for x in c0_data: stats.update(x, 0, 1.0)
        for x in c1_data: stats.update(x, 1, 1.0)

        x = np.array([10.0, -10.0])
        prob = cu.naive_bayes_prob(x, stats)
        log_prob = cu.naive_bayes_log_prob(x, stats)
        self.assertEqual(np.argmax(prob), 0)
        self.assertEqual(np.argmax(log_prob), 0)
        np.testing.assert_almost_equal(log_prob, prob, decimal=10)

        x = np.array([10.0, 0.0])
        prob = cu.naive_bayes_prob(x, stats)
        log_prob = cu.naive_bayes_log_prob(x, stats)
        self.assertEqual(np.argmax(prob), 0)
        self.assertEqual(np.argmax(log_prob), 0)
        np.testing.assert_almost_equal(log_prob, prob, decimal=10)

        x = np.array([0.0, 10.0])
        prob = cu.naive_bayes_prob(x, stats)
        log_prob = cu.naive_bayes_log_prob(x, stats)
        self.assertEqual(np.argmax(prob), 1)
        self.assertEqual(np.argmax(log_prob), 1)
        np.testing.assert_almost_equal(log_prob, prob, decimal=10)

        x = np.array([-10.0, 10.0])
        prob = cu.naive_bayes_prob(x, stats)
        log_prob = cu.naive_bayes_log_prob(x, stats)
        self.assertEqual(np.argmax(prob), 1)
        self.assertEqual(np.argmax(log_prob), 1)
        np.testing.assert_almost_equal(log_prob, prob, decimal=10)

