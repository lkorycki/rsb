import unittest
import numpy as np

from utils.stat_utils import GaussianEstimator
from data.data_utils import IndexDataset, DataUtils
from test.test_stream import DummyDataset
from utils.coll_utils import CollectionUtils as clu


class ClassificationUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_ensure_arr_size(self):
        arr = np.array([3.0, 2.0, 1.0])
        arr = clu.ensure_arr_size(arr, 4)
        np.testing.assert_equal(arr, [3.0, 2.0, 1.0, 0.0, 0.0])

        arr = np.array([3.0, 2.0, 1.0])
        arr = clu.ensure_arr_size(arr, 2)
        np.testing.assert_equal(arr, [3.0, 2.0, 1.0])

        arr = np.array([])
        arr = clu.ensure_arr_size(arr, 2)
        np.testing.assert_equal(arr, [0.0, 0.0, 0.0])

        arr = np.array([
            [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ])
        arr = clu.ensure_arr_size(arr, 4)
        np.testing.assert_equal(arr, [
            [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ])

        arr = np.array([
            [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ])
        arr = clu.ensure_arr_size(arr, 1)
        np.testing.assert_equal(arr, [
            [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ])

    def test_ensure_list2d_size(self):
        lst = [[GaussianEstimator(1), GaussianEstimator(2), GaussianEstimator(3)]]
        lst = clu.ensure_list2d_size(lst, 3, GaussianEstimator)
        self.assertEqual(len(lst), 1)
        self.assertEqual(len(lst[0]), 4)
        for i, l in enumerate(lst[0][:3]): self.assertEqual(l.get_count(), i + 1)
        self.assertEqual(lst[0][-1].get_count(), 0)

        lst = [[GaussianEstimator(1), GaussianEstimator(2), GaussianEstimator(3)]]
        lst = clu.ensure_list2d_size(lst, 2, GaussianEstimator)
        for i, l in enumerate(lst[0][:3]): self.assertEqual(l.get_count(), i + 1)

        lst = [
            [GaussianEstimator(1), GaussianEstimator(2), GaussianEstimator(3)],
            [GaussianEstimator(4), GaussianEstimator(5)]]
        lst = clu.ensure_list2d_size(lst, 4, GaussianEstimator)
        self.assertEqual(len(lst), 2)
        self.assertEqual(len(lst[0]), 5)
        self.assertEqual(len(lst[1]), 5)
        for i, l in enumerate(lst[0][:3]): self.assertEqual(l.get_count(), i + 1)
        for i, l in enumerate(lst[0][3:]): self.assertEqual(l.get_count(), 0)
        for i, l in enumerate(lst[1][2:]): self.assertEqual(l.get_count(), 0)

    def test_get_class_indices(self):
        inputs, labels = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]), [2, 3, 0, 1, 2, 2]
        dataset = DummyDataset(inputs, labels)
        class_indices = DataUtils.get_class_indices(IndexDataset(dataset))
        self.assertDictEqual(class_indices, {0: [2], 1: [3], 2: [0, 4, 5], 3: [1]})

    def test_split_list(self):
        self.assertListEqual(clu.split_list([1, 2, 3], 1), [[1], [2], [3]])
        self.assertListEqual(clu.split_list([1, 2, 3, 4, 5, 6], 2), [[1, 2], [3, 4], [5, 6]])
        self.assertListEqual(clu.split_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertListEqual(clu.split_list([1, 2, 3, 4, 5], 6), [[1, 2, 3, 4, 5]])

    def test_flatten_list(self):
        self.assertListEqual(clu.flatten_list([[1, 2, 3], [4, 5]]), [1, 2, 3, 4, 5])
        self.assertListEqual(clu.flatten_list([[[1, 2, 3], [4, 5]]]), [[1, 2, 3], [4, 5]])

