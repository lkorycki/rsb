import unittest
from collections import Counter

import torch
import numpy as np
from torch import Tensor
from pprint import pprint

from learners.er import ExperienceReplay, ClassBuffer, SubspaceBuffer, ReactiveSubspaceBuffer


class ExperienceReplayTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_class_buffer(self):
        num_atts, num_cls, batch_size = 10, 3, 10
        c0_data = np.random.multivariate_normal(np.ones(num_atts) * 10.0, np.eye(num_atts) * 1.0, batch_size)
        c1_data = np.random.multivariate_normal(np.ones(num_atts) * -10.0, np.eye(num_atts) * 1.0, batch_size)
        c2_data = np.random.multivariate_normal(np.ones(num_atts) * 0.0, np.eye(num_atts) * 1.0, batch_size)

        class_max_size = 5
        class_buffer = ClassBuffer(class_max_size)
        class_buffer.add(c0_data, np.zeros(batch_size), np.ones(batch_size))
        class_buffer.add(c1_data, np.ones(batch_size), np.ones(batch_size))
        class_buffer.add(c2_data, 2 * np.ones(batch_size), np.ones(batch_size))
        self.assertEqual(sum([len(b) for b in class_buffer.buffers.values()]), num_cls * class_max_size)

        x_batch, y_batch, weights = c0_data, np.zeros(batch_size), np.ones(batch_size)
        sampled_x_batch, sampled_y_batch, sampled_weights = class_buffer.sample(x_batch, y_batch, weights)
        self.assertListEqual(list(sampled_x_batch.shape), [len(x_batch) * (num_cls - 1), num_atts])
        self.assertEqual(len(sampled_y_batch), len(x_batch) * (num_cls - 1))
        self.assertEqual(len(sampled_weights), len(x_batch) * (num_cls - 1))

        num_cls = 2
        c0_data = np.array([
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]]
        ])
        c1_data = np.array([
            [[100.0, 200.0], [300.0, 400.0]],
            [[500.0, 600.0], [700.0, 800.0]]
        ])

        class_buffer = ClassBuffer(class_max_size)
        class_buffer.add(c0_data, np.zeros(batch_size), np.ones(batch_size))
        class_buffer.add(c1_data, np.ones(batch_size), np.ones(batch_size))
        self.assertEqual(sum([len(b) for b in class_buffer.buffers.values()]), 5)

        x_batch, y_batch, weights = c0_data, np.zeros(batch_size), np.ones(batch_size)
        sampled_x_batch, sampled_y_batch, sampled_weights = class_buffer.sample(x_batch, y_batch, weights)
        self.assertListEqual(list(sampled_x_batch.shape), [len(x_batch) * (num_cls - 1), 2, 2])
        self.assertEqual(len(sampled_y_batch), len(x_batch) * (num_cls - 1))
        self.assertEqual(len(sampled_weights), len(x_batch) * (num_cls - 1))

    def test_subspace_buffer(self):
        num_atts, num_cls, batch_size = 10, 3, 20
        c0_data = np.vstack([np.random.multivariate_normal(np.ones(num_atts) * 10.0, np.eye(num_atts) * 1.0, batch_size),
                                 np.random.multivariate_normal(np.ones(num_atts) * 20.0, np.eye(num_atts) * 1.0, batch_size)])
        c1_data = np.vstack([np.random.multivariate_normal(np.ones(num_atts) * -100.0, np.eye(num_atts) * 1.0, batch_size),
                                 np.random.multivariate_normal(np.ones(num_atts) * -200.0, np.eye(num_atts) * 1.0, batch_size)])
        c2_data = np.vstack([np.random.multivariate_normal(np.ones(num_atts) * 100.0, np.eye(num_atts) * 1.0, batch_size),
                                 np.random.multivariate_normal(np.ones(num_atts) * 200.0, np.eye(num_atts) * 1.0, batch_size)])
        np.random.shuffle(c0_data), np.random.shuffle(c1_data), np.random.shuffle(c2_data)

        max_centroids, max_instances = 2, 5
        subspace_buffer = SubspaceBuffer(max_centroids, max_instances)
        subspace_buffer.add(c0_data, np.zeros(batch_size), np.ones(batch_size))
        subspace_buffer.add(c1_data, np.ones(batch_size), np.ones(batch_size))
        subspace_buffer.add(c2_data, 2 * np.ones(batch_size), np.ones(batch_size))

        self.assertEqual(len(subspace_buffer.centroids.keys()), num_cls)
        for centroids in subspace_buffer.centroids.values():
            self.assertListEqual(list(np.array(centroids, dtype=object).shape), [max_centroids, 2])

        self.assertEqual(len(subspace_buffer.buffers.keys()), num_cls)
        for buffer in subspace_buffer.buffers.values():
            self.assertListEqual(list(np.array(buffer, dtype=object).shape), [max_centroids, max_instances, 3])

        x_batch, y_batch, weights = c0_data, np.zeros(batch_size), np.ones(batch_size)
        sampled_x_batch, sampled_y_batch, sampled_weights = subspace_buffer.sample(x_batch, y_batch, weights)
        self.assertListEqual(list(sampled_x_batch.shape), [len(x_batch) * (num_cls * max_centroids), num_atts])
        self.assertEqual(len(sampled_y_batch), len(x_batch) * (num_cls * max_centroids))
        self.assertEqual(len(sampled_weights), len(x_batch) * (num_cls * max_centroids))

    def test_reactive_subspace_buffer(self):
        max_centroids, max_instances, window_size = 2, 5, 3
        reactive_buffer = ReactiveSubspaceBuffer(max_centroids, max_instances, window_size)

        cls_1, cls_3 = 1, 3
        ctr_1_idx, ctr_3_1_idx, ctr_3_2_idx = 0, 1, 2
        t1, t3_1, t3_2 = (np.array([1, 1, 1]), cls_1, 2.0), (np.array([10, 10, 10]), cls_3, 1.0), (np.array([50, 50, 50]), cls_3, 1.0)
        reactive_buffer._ReactiveSubspaceBuffer__add_centroid(*t1)
        reactive_buffer._ReactiveSubspaceBuffer__add_centroid(*t3_1)
        reactive_buffer._ReactiveSubspaceBuffer__add_centroid(*t3_2)

        self.__test_buffers(t1, t3_1, t3_2, cls_1, cls_3, ctr_1_idx, ctr_3_1_idx, ctr_3_2_idx, reactive_buffer.buffers)
        self.__test_buffers(t1, t3_1, t3_2, cls_1, cls_3, ctr_1_idx, ctr_3_1_idx, ctr_3_2_idx, reactive_buffer.centroids_window_buffers)

        self.assertEqual(len(reactive_buffer.centroids), 2)
        self.assertEqual(len(reactive_buffer.centroids[cls_1]), 1)
        centroid = reactive_buffer.centroids[cls_1][ctr_1_idx]
        self.assertListEqual(centroid[0].tolist(), t1[0].tolist())
        self.assertEqual(centroid[1], t1[-1])
        self.assertListEqual(centroid[2].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(len(reactive_buffer.centroids[cls_3]), 2)
        centroid = reactive_buffer.centroids[cls_3][ctr_3_1_idx]
        self.assertListEqual(centroid[0].tolist(), t3_1[0].tolist())
        self.assertEqual(centroid[1], t3_1[-1])
        self.assertListEqual(centroid[2].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(reactive_buffer.total_num_centroids, 3)
        centroid = reactive_buffer.centroids[cls_3][ctr_3_2_idx]
        self.assertListEqual(centroid[0].tolist(), t3_2[0].tolist())
        self.assertEqual(centroid[1], t3_2[-1])
        self.assertListEqual(centroid[2].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(reactive_buffer.total_num_centroids, 3)

        self.assertEqual(len(reactive_buffer.centroids_window_counts), 2)
        self.assertEqual(len(reactive_buffer.centroids_window_counts[cls_1]), 1)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_1][ctr_1_idx], {1: 1})
        self.assertEqual(len(reactive_buffer.centroids_window_counts[cls_3]), 2)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_3][ctr_3_1_idx], {3: 1})
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_3][ctr_3_2_idx], {3: 1})

        self.assertEqual(reactive_buffer.next_centroid_idx, 3)

        closest_centroid_idx, closest_centroid_y, dist = reactive_buffer._ReactiveSubspaceBuffer__find_closest_centroid(np.array([15, 15, 15]))
        self.assertEqual(closest_centroid_idx, ctr_3_1_idx)
        self.assertEqual(closest_centroid_y, cls_3)
        self.assertAlmostEqual(dist, 8.66, delta=0.01)
        closest_centroid_idx, closest_centroid_y, dist = reactive_buffer._ReactiveSubspaceBuffer__find_closest_centroid(np.array([-1, -1, -1]))
        self.assertEqual(closest_centroid_idx, ctr_1_idx)
        self.assertEqual(closest_centroid_y, cls_1)
        self.assertAlmostEqual(dist, 3.46, delta=0.01)
        closest_centroid_idx, closest_centroid_y, dist = reactive_buffer._ReactiveSubspaceBuffer__find_closest_centroid(np.array([0.0, 0.0, 0.0]), cls_3)
        self.assertEqual(closest_centroid_idx, ctr_3_1_idx)
        self.assertEqual(closest_centroid_y, cls_3)
        self.assertAlmostEqual(dist, 17.32, delta=0.01)

        centroid_idx = ctr_3_1_idx
        reactive_buffer._ReactiveSubspaceBuffer__update_centroid_window(np.array([11, 11, 11]), 3, 1.0, cls_3, centroid_idx)
        reactive_buffer._ReactiveSubspaceBuffer__update_centroid_window(np.array([4, 4, 4]), 1, 1.0, cls_3, centroid_idx)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[cls_3][centroid_idx]), 3)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_3][centroid_idx], {1: 1, 3: 2})

        reactive_buffer._ReactiveSubspaceBuffer__update_centroid_window(np.array([5, 5, 5]), 1, 1.0, cls_3, centroid_idx)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[cls_3][centroid_idx]), 3)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_3][centroid_idx], {1: 2, 3: 1})

        centroid_idx = ctr_3_2_idx
        reactive_buffer._ReactiveSubspaceBuffer__update_centroid(np.array([100, 100, 100]), 3, 1.0, cls_3, centroid_idx)
        centroid = reactive_buffer.centroids[cls_3][centroid_idx]
        self.assertListEqual(centroid[0].tolist(), [75.0, 75.0, 75.0])
        self.assertEqual(centroid[1], 2.0)
        self.assertListEqual(centroid[2].tolist(), [1250.0, 1250.0, 1250.0])
        self.assertEqual(len(reactive_buffer.buffers[cls_3][centroid_idx]), 2)

        for _ in range(10): reactive_buffer._ReactiveSubspaceBuffer__update_centroid(np.array([100, 100, 100]), 3, 1.0, cls_3, centroid_idx)
        self.assertEqual(len(reactive_buffer.buffers[cls_3][centroid_idx]), max_instances)

        self.assertSequenceEqual(reactive_buffer._ReactiveSubspaceBuffer__check_centroid_switch(cls_1, ctr_1_idx), (False, cls_1))
        self.assertSequenceEqual(reactive_buffer._ReactiveSubspaceBuffer__check_centroid_switch(cls_3, ctr_3_1_idx), (True, cls_1))

        ctr_1_2_idx = 3
        reactive_buffer._ReactiveSubspaceBuffer__switch_centroid(cls_3, ctr_3_1_idx, cls_1)
        self.assertTrue(ctr_1_2_idx in reactive_buffer.centroids[cls_1])
        self.assertListEqual(reactive_buffer.centroids[cls_1][ctr_1_2_idx][0].tolist(), [4.5, 4.5, 4.5])
        self.assertEqual(reactive_buffer.centroids[cls_1][ctr_1_2_idx][1], 2.0)
        self.assertListEqual(reactive_buffer.centroids[cls_1][ctr_1_2_idx][2].tolist(), [0.5, 0.5, 0.5])
        self.assertTrue(ctr_1_2_idx in reactive_buffer.centroids_window_counts[cls_1])
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_1][ctr_1_2_idx], {1: 2, 3: 1})
        self.assertTrue(ctr_1_2_idx in reactive_buffer.centroids_window_buffers[cls_1])
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[cls_1][ctr_1_2_idx]), 3)
        self.assertTrue(ctr_1_2_idx in reactive_buffer.buffers[cls_1])

        self.assertTrue(ctr_3_1_idx not in reactive_buffer.centroids[cls_3])
        self.assertTrue(ctr_3_1_idx not in reactive_buffer.centroids_window_counts[cls_3])
        self.assertTrue(ctr_3_1_idx not in reactive_buffer.centroids_window_buffers[cls_3])
        self.assertTrue(ctr_3_1_idx not in reactive_buffer.buffers[cls_3])
        self.assertEqual(reactive_buffer.total_num_centroids, 3)
        self.assertEqual(reactive_buffer.next_centroid_idx, 4)

        num_cls, num_atts, total_num_centroids, pure_centroids = 2, 3, 3, 2
        x_batch, y_batch, weights = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]), np.zeros(3), np.ones(3)
        sampled_x_batch, sampled_y_batch, sampled_weights = reactive_buffer.sample(x_batch, y_batch, weights)
        self.assertGreaterEqual(sampled_x_batch.shape[0], len(x_batch) * pure_centroids)
        self.assertLessEqual(sampled_x_batch.shape[0], len(x_batch) * num_cls * total_num_centroids)
        self.assertEqual(sampled_x_batch.shape[1], num_atts)
        self.assertGreaterEqual(len(sampled_y_batch), len(x_batch) * pure_centroids)
        self.assertLessEqual(len(sampled_y_batch), len(x_batch) * num_cls * total_num_centroids)
        self.assertGreaterEqual(len(sampled_weights), len(x_batch) * pure_centroids)
        self.assertLessEqual(len(sampled_weights), len(x_batch) * num_cls * total_num_centroids)

        max_centroids, max_instances, window_size = 2, 10, 10
        reactive_buffer = ReactiveSubspaceBuffer(max_centroids, max_instances, window_size, split_thresh=0.5, split_period=0)

        ctr_idx, ctr_5_idx, cls_5, ctr_6_idx, cls_6 = 0, 1, 5, 2, 6
        t5 = (np.array([500, 500, 500]), cls_5, 1.0)
        t6 = (np.array([600, 600, 600]), cls_6, 1.0)

        reactive_buffer._ReactiveSubspaceBuffer__add_centroid(*t5)
        for _ in range(6):
            reactive_buffer._ReactiveSubspaceBuffer__update_centroid_window(*t5, cls_5, ctr_idx)
            reactive_buffer._ReactiveSubspaceBuffer__update_centroid(*t5, cls_5, ctr_idx)
        for _ in range(4):
            reactive_buffer._ReactiveSubspaceBuffer__update_centroid_window(*t6, cls_5, ctr_idx)
            reactive_buffer._ReactiveSubspaceBuffer__update_centroid(*t6, cls_5, ctr_idx)

        reactive_buffer._ReactiveSubspaceBuffer__check_centroids()

        self.assertEqual(len(reactive_buffer.centroids), 2)
        self.assertEqual(reactive_buffer.total_num_centroids, 2)
        self.assertEqual(len(reactive_buffer.centroids[cls_5]), 1)
        self.assertEqual(len(reactive_buffer.centroids[cls_6]), 1)

        centroid = reactive_buffer.centroids[cls_5][ctr_5_idx]
        self.assertListEqual(centroid[0].tolist(), t5[0].tolist())
        self.assertEqual(centroid[1], 6 * t5[-1])
        self.assertListEqual(centroid[2].tolist(), [0.0, 0.0, 0.0])
        centroid = reactive_buffer.centroids[cls_6][ctr_6_idx]
        self.assertListEqual(centroid[0].tolist(), t6[0].tolist())
        self.assertEqual(centroid[1], 4 * t6[-1])
        self.assertListEqual(centroid[2].tolist(), [0.0, 0.0, 0.0])

        self.assertEqual(len(reactive_buffer.centroids_window_buffers), 2)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[cls_5][ctr_5_idx]), 6)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[cls_6][ctr_6_idx]), 4)

        self.assertEqual(len(reactive_buffer.centroids_window_counts), 2)
        self.assertEqual(len(reactive_buffer.centroids_window_counts[cls_5]), 1)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_5][ctr_5_idx], {5: 6})
        self.assertEqual(len(reactive_buffer.centroids_window_counts[cls_6]), 1)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[cls_6][ctr_6_idx], {6: 4})

    def test_reactive_subspace_multidim_buffer(self):
        max_centroids, max_instances, window_size = 1, 10, 10
        reactive_buffer = ReactiveSubspaceBuffer(max_centroids, max_instances, window_size)

        c0_data_1ch = np.array([
            [
                [
                    [1, 1, 1],
                    [10, 10, 10],
                    [100, 100, 100]
                ]
            ], [
                [
                    [5, 5, 5],
                    [50, 50, 50],
                    [500, 500, 500]
                ]
            ]
        ])

        reactive_buffer.add(c0_data_1ch, np.zeros(len(c0_data_1ch)), np.ones(len(c0_data_1ch)))

        self.assertEqual(len(reactive_buffer.centroids), 1)
        self.assertEqual(reactive_buffer.total_num_centroids, 1)
        self.assertEqual(len(reactive_buffer.centroids[0]), 1)
        self.assertEqual(len(reactive_buffer.centroids[0]), 1)

        centroid = reactive_buffer.centroids[0][0]
        self.assertListEqual(centroid[0].tolist(), [[[3.0, 3.0, 3.0], [30.0, 30.0, 30.0], [300.0, 300.0, 300.0]]])
        self.assertEqual(centroid[1], 2.0)
        self.assertListEqual(centroid[2].tolist(), [[[8.0, 8.0, 8.0], [800.0, 800.0, 800.0], [80000.0, 80000.0, 80000.0]]])

        self.assertEqual(len(reactive_buffer.centroids_window_buffers), 1)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[0][0]), 2)

        self.assertEqual(len(reactive_buffer.centroids_window_counts), 1)
        self.assertEqual(len(reactive_buffer.centroids_window_counts[0]), 1)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[0][0], {0: 2})

        c0_data_3ch = np.array([
            [
                [
                    [1, 1, 1],
                    [10, 10, 10],
                    [100, 100, 100]
                ], [
                    [2, 2, 2],
                    [20, 20, 20],
                    [200, 200, 200]
                ], [
                    [3, 3, 3],
                    [30, 30, 30],
                    [300, 300, 300]
                ]
            ], [
                [
                    [5, 5, 5],
                    [50, 50, 50],
                    [500, 500, 500]
                ], [
                    [6, 6, 6],
                    [60, 60, 60],
                    [600, 600, 600]
                ], [
                    [7, 7, 7],
                    [70, 70, 70],
                    [700, 700, 700]
                ]
            ]
        ])

        reactive_buffer = ReactiveSubspaceBuffer(max_centroids, max_instances, window_size)

        reactive_buffer.add(c0_data_3ch, np.zeros(len(c0_data_3ch)), np.ones(len(c0_data_3ch)))

        self.assertEqual(len(reactive_buffer.centroids), 1)
        self.assertEqual(reactive_buffer.total_num_centroids, 1)
        self.assertEqual(len(reactive_buffer.centroids[0]), 1)
        self.assertEqual(len(reactive_buffer.centroids[0]), 1)

        centroid = reactive_buffer.centroids[0][0]
        self.assertListEqual(centroid[0].tolist(), [
            [
                [3.0, 3.0, 3.0],
                [30.0, 30.0, 30.0],
                [300.0, 300.0, 300.0]
            ], [
                [4.0, 4.0, 4.0],
                [40.0, 40.0, 40.0],
                [400.0, 400.0, 400.0]
            ], [
                [5.0, 5.0, 5.0],
                [50.0, 50.0, 50.0],
                [500.0, 500.0, 500.0]
            ]
        ])
        self.assertEqual(centroid[1], 2.0)
        self.assertListEqual(centroid[2].tolist(), [
            [
                [8.0, 8.0, 8.0],
                [800.0, 800.0, 800.0],
                [80000.0, 80000.0, 80000.0]
            ], [
                [8.0, 8.0, 8.0],
                [800.0, 800.0, 800.0],
                [80000.0, 80000.0, 80000.0]
            ], [
                [8.0, 8.0, 8.0],
                [800.0, 800.0, 800.0],
                [80000.0, 80000.0, 80000.0]
            ]
        ])

        self.assertEqual(len(reactive_buffer.centroids_window_buffers), 1)
        self.assertEqual(len(reactive_buffer.centroids_window_buffers[0][0]), 2)

        self.assertEqual(len(reactive_buffer.centroids_window_counts), 1)
        self.assertEqual(len(reactive_buffer.centroids_window_counts[0]), 1)
        self.assertDictEqual(reactive_buffer.centroids_window_counts[0][0], {0: 2})

        reactive_buffer.add(np.array([c0_data_3ch[0]]), np.ones(1), np.ones(1))

        closest_centroid_idx, closest_centroid_y, dist = reactive_buffer._ReactiveSubspaceBuffer__find_closest_centroid(c0_data_3ch[0])
        self.assertEqual(closest_centroid_idx, 1)
        self.assertEqual(closest_centroid_y, 1)
        self.assertAlmostEqual(dist, 0.0, delta=0.01)
        closest_centroid_idx, closest_centroid_y, dist = reactive_buffer._ReactiveSubspaceBuffer__find_closest_centroid(2 * c0_data_3ch[1])
        self.assertEqual(closest_centroid_idx, 0)
        self.assertEqual(closest_centroid_y, 0)
        self.assertAlmostEqual(dist, 2424.61, delta=0.01)

    def __test_buffers(self, t1, t3_1, t3_2, cls_1, cls_3, ctr_1_idx, ctr_3_1_idx, ctr_3_2_idx, buffers):
        self.assertEqual(len(buffers), 2)
        self.assertEqual(len(buffers[cls_1]), 1)
        buffer = buffers[cls_1][ctr_1_idx]
        self.assertListEqual(buffer[0][0].tolist(), t1[0].tolist())
        self.assertEqual(buffer[0][1], t1[1])
        self.assertEqual(buffer[0][2], t1[2])
        self.assertEqual(len(buffers[cls_3]), 2)
        buffer = buffers[cls_3][ctr_3_1_idx]
        self.assertListEqual(buffer[0][0].tolist(), t3_1[0].tolist())
        self.assertEqual(buffer[0][1], t3_1[1])
        self.assertEqual(buffer[0][2], t3_1[2])
        buffer = buffers[cls_3][ctr_3_2_idx]
        self.assertListEqual(buffer[0][0].tolist(), t3_2[0].tolist())
        self.assertEqual(buffer[0][1], t3_2[1])
        self.assertEqual(buffer[0][2], t3_2[2])

    def test_rb_resample(self):
        max_cls, num_cls = 4, 2
        x_batch, y_batch, weights = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), np.array([1, 1, 0, 1, 1]), np.ones(5)
        cls_indices = {0: [2], 1: [0, 1, 3, 4]}
        rb = ReactiveSubspaceBuffer(1, 1)

        resampled_x_batch, resampled_y_batch, resampled_weights = rb._ReactiveSubspaceBuffer__resample(x_batch, y_batch, weights, cls_indices)
        self.assertEqual(len(resampled_x_batch), num_cls * max_cls)
        self.assertEqual(len(resampled_y_batch), num_cls * max_cls)
        self.assertDictEqual(dict(Counter(resampled_y_batch)), {0: 4, 1: 4})
        self.assertEqual(len(resampled_weights), num_cls * max_cls)

        max_cls, num_cls = 3, 3
        x_batch, y_batch, weights = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]), np.array([1, 1, 0, 1, 2, 2]), np.ones(6)
        cls_indices = {0: [2], 1: [0, 1, 3], 2: [4, 5]}
        rb = ReactiveSubspaceBuffer(1, 1)

        resampled_x_batch, resampled_y_batch, resampled_weights = rb._ReactiveSubspaceBuffer__resample(x_batch, y_batch, weights, cls_indices)
        self.assertEqual(len(resampled_x_batch), num_cls * max_cls)
        self.assertEqual(len(resampled_y_batch), num_cls * max_cls)
        self.assertDictEqual(dict(Counter(resampled_y_batch)), {0: 3, 1: 3, 2: 3})
        self.assertEqual(len(resampled_weights), num_cls * max_cls)

    def test_er(self):
        input_batch = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]), Tensor([0, 1, 0]), torch.ones(3)
        sampled_batch = Tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]), Tensor([1, 0]), torch.ones(2)

        ext_x_batch, ext_y_batch, ext_weights = ExperienceReplay.extend(input_batch, sampled_batch)

        np.testing.assert_equal(ext_x_batch.numpy(), np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0],
                                                               [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]))
        np.testing.assert_equal(ext_y_batch.numpy(), np.array([0, 1, 0, 1, 0]))
        np.testing.assert_equal(ext_weights.numpy(), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))

        input_batch = (Tensor([
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]]
        ]), Tensor([0, 1, 0]), torch.ones(3))

        sampled_batch = (Tensor([
            [[100.0, 200.0], [300.0, 400.0]],
            [[500.0, 600.0], [700.0, 800.0]]
        ]), Tensor([1, 0]), torch.ones(2))

        ext_x_batch, ext_y_batch, ext_weights = ExperienceReplay.extend(input_batch, sampled_batch)

        np.testing.assert_equal(ext_x_batch.numpy(), np.array([
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]],
            [[100.0, 200.0], [300.0, 400.0]],
            [[500.0, 600.0], [700.0, 800.0]]
        ]))
        np.testing.assert_equal(ext_y_batch.numpy(), np.array([0, 1, 0, 1, 0]))
        np.testing.assert_equal(ext_weights.numpy(), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
