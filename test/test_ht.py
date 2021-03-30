import datetime

import learners.ht as ht

import unittest
import numpy as np
import skmultiflow as sk
import math


class HoeffdingTreeTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_tree_node_split(self):
        num_atts, num_cls = 3, 2
        node = ht.TreeNode(num_atts=num_atts, num_cls=num_cls)
        self.assertTrue(node.is_leaf)
        self.assertEqual(node.num_atts, num_atts)
        self.assertListEqual(node.split_atts.tolist(), [0, 1, 2])
        self.assertEqual(node.stat_atts.tolist(), [0, 1, 2])
        self.assertIsNotNone(node.stats)

        node.attempt_split()
        self.assertTrue(node.is_leaf)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

        c0_data = np.random.multivariate_normal(np.array([4.0, -10.0, 5.0]), np.eye(num_atts) * 10.0, 100)
        c1_data = np.random.multivariate_normal(np.array([6.0, 10.0, 5.0]), np.eye(num_atts) * 10.0, 100)
        for x in c0_data: node.update(x, 0, 1.0)
        for x in c1_data: node.update(x, 1, 1.0)

        node.attempt_split()
        self.assertFalse(node.is_leaf)
        self.assertEqual(node.split[0], 1)
        self.assertAlmostEqual(node.split[0], 0.0, delta=2.0)

        left_node = node.left
        right_node = node.right
        self.assertIsNotNone(left_node)
        self.assertIsNotNone(right_node)
        self.assertTrue(left_node.is_leaf)
        self.assertTrue(right_node.is_leaf)

        self.assertIsNotNone(left_node.stats)
        self.assertAlmostEqual(left_node.stats.get_cls_count(0), 100, delta=1.0)
        self.assertAlmostEqual(left_node.stats.get_cls_count(1), 0, delta=1.0)
        self.assertIsNotNone(right_node.stats)
        self.assertAlmostEqual(right_node.stats.get_cls_count(0), 0, delta=1.0)
        self.assertAlmostEqual(right_node.stats.get_cls_count(1), 100, delta=1.0)

        node = node.left
        c1_new_data = np.random.multivariate_normal(np.array([20.0, -10.0, 5.0]), np.eye(num_atts) * 10.0, 100)
        for x in c0_data: node.update(x, 0, 1.0)
        for x in c1_new_data: node.update(x, 1, 1.0)

        node.attempt_split()
        self.assertFalse(node.is_leaf)
        self.assertEqual(node.split[0], 0)
        self.assertAlmostEqual(node.split[1], 12.0, delta=2.0)

    def test_random_tree_node_split(self):
        num_atts, num_cls = 100, 10
        num_rnd_atts = math.sqrt(num_atts) + 1
        tree = ht.HoeffdingTree(rnd=True, num_atts=num_atts, num_cls=num_cls)
        root = tree.root

        self.assertEqual(len(root.split_atts), num_rnd_atts)
        self.assertEqual(len(root.stat_atts), num_rnd_atts)
        self.assertEqual(root.num_atts, num_atts)
        self.assertListEqual(root.split_atts.tolist(), root.stat_atts.tolist())

        tree = ht.HoeffdingTree(rnd=True)
        root = tree.root
        c0_data = np.random.multivariate_normal(10 * np.ones(num_atts), np.eye(num_atts) * 10.0, 100)
        c1_data = np.random.multivariate_normal(-10 * np.ones(num_atts), np.eye(num_atts) * 10.0, 100)
        for x in c0_data: root.update(x, 0, 1.0)
        for x in c1_data: root.update(x, 1, 1.0)
        root.attempt_split()
        left_node = root.left
        right_node = root.right

        self.assertIsNotNone(left_node)
        self.assertEqual(len(left_node.split_atts), num_rnd_atts)
        self.assertEqual(len(left_node.stat_atts), num_rnd_atts)
        self.assertEqual(left_node.num_atts, num_atts)
        self.assertListEqual(left_node.split_atts.tolist(), left_node.stat_atts.tolist())

        self.assertIsNotNone(right_node)
        self.assertEqual(len(right_node.split_atts), num_rnd_atts)
        self.assertEqual(len(right_node.stat_atts), num_rnd_atts)
        self.assertEqual(right_node.num_atts, num_atts)
        self.assertListEqual(right_node.split_atts.tolist(), right_node.stat_atts.tolist())

        tree = ht.HoeffdingTree(rnd=True, att_split_est=True, num_atts=num_atts, num_cls=num_cls)
        root = tree.root

        self.assertEqual(len(root.split_atts), num_rnd_atts)
        self.assertEqual(len(root.stat_atts), num_atts)

    def test_hoeffding_tree_routing(self):
        num_atts, num_cls = 3, 2
        tree = ht.HoeffdingTree()

        leaf = tree.find_leaf(np.array([0.0, 0.0, 0.0]))
        self.assertIsNotNone(leaf)
        self.assertEqual(leaf.cnt, 0.0)

        leaf.split = (1, 10.0)
        leaf.left = ht.TreeNode(num_atts, num_cls)
        leaf.left.cnt = 100.0
        leaf.right = ht.TreeNode(num_atts, num_cls)
        leaf.right.cnt = 200.0
        leaf.is_leaf = False

        leaf = tree.find_leaf(np.array([15.0, 5.0, 0.0]))
        self.assertIsNotNone(leaf)
        self.assertEqual(leaf.cnt, 100.0)

        leaf = tree.find_leaf(np.array([5.0, 15.0, 0.0]))
        self.assertIsNotNone(leaf)
        self.assertEqual(leaf.cnt, 200.0)

    def test_hoeffding_tree_pred(self):
        tree = ht.HoeffdingTree()

        probs = tree.predict_prob(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
        self.assertEqual(probs.shape, (2, 1))
        self.assertListEqual(probs[0].tolist(), [0.0])

        preds = tree.predict(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
        self.assertEqual(preds.shape, (2,))
        self.assertEqual(preds[0], 0.0)
        self.assertEqual(preds[1], 0.0)

        tree.update(np.random.randn(5, 3), np.ones(5, dtype=np.int))
        probs = tree.predict_prob(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
        self.assertEqual(probs.shape, (2, 2))

    def test_hoeffding_tree_update(self):
        tree = ht.HoeffdingTree()

        num_atts, num_cls = 3, 2
        tree.update(np.random.randn(5, num_atts), np.zeros(5, dtype=np.int), weights=np.ones(5) * 2)
        leaf = tree.root
        self.assertEqual(leaf.cnt, 10.0)
        self.assertEqual(leaf.num_cls, 1)
        self.assertEqual(leaf.num_atts, 3)

        tree.update(np.random.randn(5, num_atts), np.ones(5, dtype=np.int), weights=np.ones(5) * 2)
        self.assertEqual(leaf.cnt, 20.0)
        self.assertEqual(leaf.num_cls, 2)
        self.assertEqual(leaf.num_atts, 3)

    @unittest.skip
    def test_ht_perf(self):
        num_atts, num_cls, = 1000, 2
        c0_data = np.random.multivariate_normal(np.ones(num_atts) * 10.0, np.eye(num_atts) * 10.0, 150)
        c1_data = np.random.multivariate_normal(np.ones(num_atts) * -10.0, np.eye(num_atts) * 10.0, 150)

        ht1 = ht.HoeffdingTree()
        #ht2 = sk.trees.HoeffdingTreeClassifier()
        start_time = datetime.datetime.now()
        ht1.update(c0_data, np.zeros(len(c0_data)), weights=np.ones(len(c0_data)))
        #ht2.update(c0_data, np.zeros(len(c0_data)), [0, 1])
        print(datetime.datetime.now() - start_time)
