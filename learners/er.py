import collections
import copy
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from operator import itemgetter
from pprint import pprint

import numpy as np
import torch
from torch import Tensor

from core.clearn import ContinualLearner
from learners.nnet import NeuralNet


class ReplayBuffer(ABC):

    @abstractmethod
    def add(self, x_batch, y_batch, weights, **kwargs):
        pass

    @abstractmethod
    def sample(self, x_batch, y_batch, weights, **kwargs):
        pass


class ClassBuffer(ReplayBuffer):

    def __init__(self, buffer_max_size: int, class_frac: float=1.0, replace_prob: float=0.5):
        self.buffer_max_size = buffer_max_size
        self.class_frac = class_frac
        self.replace_prob = replace_prob

        self.buffers = collections.defaultdict(list)

    def add(self, x_batch, y_batch, weights, **kwargs):
        if torch.is_tensor(x_batch):
            x_batch, y_batch, weights = x_batch.numpy(), y_batch.numpy(), weights.numpy()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.buffers[y]) == self.buffer_max_size:
                if self.replace_prob > random.random():
                    r = random.choice(range(len(self.buffers[y])))
                    self.buffers[y][r] = (x, y, w)
            else:
                self.buffers[y].append((x, y, w))

    def sample(self, x_batch, y_batch, weights, **kwargs):
        num_samples_per_instance = int(self.class_frac * (len(self.buffers) - 1))
        num_samples = num_samples_per_instance * len(x_batch)

        input_shape = x_batch[0].shape
        sampled_x_batch = np.zeros((num_samples, *input_shape))
        sampled_y_batch = np.zeros(num_samples)
        sampled_weights = np.zeros(num_samples)

        i = 0
        for x, y in zip(x_batch, y_batch):
            buffers = iter(self.buffers.items())

            j = 0
            while j < num_samples_per_instance:
                class_idx, class_buffer = next(buffers)

                if class_idx != y:
                    (rx, ry, rw) = random.choice(self.buffers[class_idx])
                    sampled_x_batch[i, :] = rx[:]
                    sampled_y_batch[i] = ry
                    sampled_weights[i] = rw
                    i += 1
                    j += 1

        return sampled_x_batch, sampled_y_batch, sampled_weights


class SubspaceBuffer(ReplayBuffer):
    def __init__(self, max_centroids: int, max_instances: int, centroids_frac: float=1.0):
        self.max_centroids = max_centroids
        self.max_instances = max_instances
        self.centroids_frac = centroids_frac

        self.centroids = collections.defaultdict(list)
        self.total_num_centroids = 0
        self.buffers = collections.defaultdict(list)

    def add(self, x_batch, y_batch, weights, **kwargs):
        if torch.is_tensor(x_batch):
            x_batch, y_batch, weights = x_batch.numpy(), y_batch.numpy(), weights.numpy()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.centroids[y]) == self.max_centroids:
                centroid_idx, _ = min([(i, np.linalg.norm(x - p[0])) for i, p in enumerate(self.centroids[y])], key=itemgetter(1))
                mean, w_sum = self.centroids[y][centroid_idx]
                self.centroids[y][centroid_idx] = (mean + (w / (w_sum + w)) * (x - mean), w_sum + w)

                if len(self.buffers[y][centroid_idx]) == self.max_instances:
                    self.buffers[y][centroid_idx].pop(0)

                self.buffers[y][centroid_idx].append((x, y, w))
            else:
                self.centroids[y].append((x, w))
                self.total_num_centroids += 1
                self.buffers[y].append([(x, y, w)])

    def sample(self, x_batch, y_batch, weights, **kwargs):
        num_samples_per_instance = int(self.centroids_frac * self.total_num_centroids)
        num_samples = num_samples_per_instance * len(x_batch)

        input_shape = x_batch[0].shape
        sampled_x_batch = np.zeros((num_samples, *input_shape))
        sampled_y_batch = np.zeros(num_samples)
        sampled_weights = np.zeros(num_samples)

        i = 0
        centroids_buffers = list(reduce(lambda a, b: a + b, self.buffers.values(), []))
        random.shuffle(centroids_buffers)

        for _ in range(len(x_batch)):
            for j in range(num_samples_per_instance):
                (rx, ry, rw) = random.choice(centroids_buffers[j])
                sampled_x_batch[i, :] = rx[:]
                sampled_y_batch[i] = ry
                sampled_weights[i] = rw
                i += 1

        return sampled_x_batch, sampled_y_batch, sampled_weights


class ReactiveSubspaceBuffer(ReplayBuffer):

    def __init__(self, max_centroids: int, max_instances: int, window_size: int=100, switch_thresh: float=0.9, split=False,
                 split_thresh: float=0.5, split_period: int=1000):
        super().__init__()
        self.max_centroids = max_centroids
        self.max_instances = max_instances
        self.window_size = window_size
        self.switch_thresh = switch_thresh
        self.split = split
        self.split_thresh = split_thresh
        self.split_period = split_period

        self.splits_num = 0
        self.switches_num = 0

        self.centroids = collections.defaultdict(lambda: collections.defaultdict(tuple))
        self.total_num_centroids = 0
        self.buffers = collections.defaultdict(lambda: collections.defaultdict(list))

        self.centroids_window_counts = collections.defaultdict(lambda: collections.defaultdict(Counter))
        self.centroids_window_buffers = collections.defaultdict(lambda: collections.defaultdict(list))
        self.centroids_window_last_update = collections.defaultdict(lambda: collections.defaultdict(int))
        self.t = 0

        self.next_centroid_idx = 0

    def add(self, x_batch, y_batch, weights, **kwargs):
        self.t += len(x_batch)

        if torch.is_tensor(x_batch):
            x_batch, y_batch, weights = x_batch.numpy(), y_batch.numpy(), weights.numpy()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.centroids[y]) < self.max_centroids / 2:
                self.__add_centroid(x, y, w)
                continue

            closest_centroid_idx, closest_centroid_y, dist = self.__find_closest_centroid(x)

            if closest_centroid_y == y:
                self.__update_centroid(x, y, w, closest_centroid_y, closest_centroid_idx)
                self.__update_centroid_window(x, y, w, closest_centroid_y, closest_centroid_idx)
            else:
                w_sum, var = self.centroids[closest_centroid_y][closest_centroid_idx][1:]
                std = math.sqrt(var.mean() / w_sum)

                if dist / math.sqrt(x.size) <= std:
                    window_buffer = self.__update_centroid_window(x, y, w, closest_centroid_y, closest_centroid_idx)

                    if len(window_buffer) == self.window_size:
                        centroid_switch, max_class = self.__check_centroid_switch(closest_centroid_y, closest_centroid_idx)
                        if centroid_switch:
                            self.switches_num += 1
                            self.__switch_centroid(closest_centroid_y, closest_centroid_idx, max_class)
                else:
                    closest_y_centroid_idx, y, dist = self.__find_closest_centroid(x, y)
                    w_sum, var = self.centroids[y][closest_y_centroid_idx][1:]
                    std = math.sqrt(var.mean() / w_sum)

                    if dist / math.sqrt(x.size) <= std or len(self.centroids_window_buffers[y][closest_y_centroid_idx]) < self.window_size \
                            or len(self.centroids[y]) >= self.max_centroids:
                        self.__update_centroid(x, y, w, y, closest_y_centroid_idx)
                        self.__update_centroid_window(x, y, w, y, closest_y_centroid_idx)
                    else:
                        self.__add_centroid(x, y, w)

        if self.split:
            self.__check_centroids()

    def __add_centroid(self, x, y, w):
        self.centroids[y][self.next_centroid_idx] = (x, w, np.zeros(len(x)))
        self.buffers[y][self.next_centroid_idx] = [(x, y, w)]
        self.centroids_window_counts[y][self.next_centroid_idx] = Counter([y])
        self.centroids_window_buffers[y][self.next_centroid_idx] = [(x, y, w)]
        self.centroids_window_last_update[y][self.next_centroid_idx] = self.t
        self.total_num_centroids += 1
        self.next_centroid_idx += 1

    def __find_closest_centroid(self, x, y=-1):
        closest_centroid_idx, closest_centroid_y, min_dist = -1, -1, float('inf')
        centroids = self.centroids.items() if y < 0 else [(y, self.centroids[y])]

        for cy, class_centroids in centroids:
            if len(class_centroids) == 0:
                continue

            centroid_idx, dist = min([(centroid_idx, np.linalg.norm(x - cv[0])) for centroid_idx, cv in class_centroids.items()], key=itemgetter(1))
            if dist < min_dist:
                closest_centroid_idx = centroid_idx
                closest_centroid_y = cy
                min_dist = dist

        return closest_centroid_idx, closest_centroid_y, min_dist

    def __update_centroid_window(self, x, y, w, centroid_y, centroid_idx):
        window_buffer = self.centroids_window_buffers[centroid_y][centroid_idx]

        if len(window_buffer) == self.window_size:
            _, wy, _ = window_buffer.pop(0)
            self.centroids_window_counts[centroid_y][centroid_idx][wy] -= 1

        window_buffer.append((x, y, w))
        self.centroids_window_counts[centroid_y][centroid_idx][y] += 1
        self.centroids_window_last_update[centroid_y][centroid_idx] = self.t

        return window_buffer

    def __update_centroid(self, x, y, w, centroid_y, centroid_idx):
        mean, w_sum, var = self.centroids[centroid_y][centroid_idx]
        new_mean = mean + (w / (w_sum + w)) * (x - mean)
        new_w_sum = w_sum + w
        new_var = var + w * np.multiply(x - mean, x - new_mean)

        self.centroids[centroid_y][centroid_idx] = (
            new_mean,
            new_w_sum,
            np.array(new_var)
        )

        if len(self.buffers[centroid_y][centroid_idx]) == self.max_instances:
            self.buffers[centroid_y][centroid_idx].pop(0)

        self.buffers[centroid_y][centroid_idx].append((x, y, w))

    def __check_centroid_switch(self, centroid_y, centroid_idx):
        max_class, max_cnt = self.centroids_window_counts[centroid_y][centroid_idx].most_common(1)[0]
        current_cls_cnt = self.centroids_window_counts[centroid_y][centroid_idx].get(centroid_y)

        return current_cls_cnt / max_cnt < self.switch_thresh, max_class

    def __switch_centroid(self, centroid_y, centroid_idx, new_class):
        self.__extract_new_centroid(centroid_y, centroid_idx, new_class)
        self.__remove_centroid(centroid_y, centroid_idx)

    def __extract_new_centroid(self, centroid_y, centroid_idx, new_class, split=False):
        window_buffer = self.centroids_window_buffers[centroid_y][centroid_idx]
        filtered_window = list(filter(lambda r: r[1] == new_class, window_buffer))

        mean, w_sum, var = filtered_window[0][2] * filtered_window[0][0], filtered_window[0][2], [0.0] * len(filtered_window[0][0])
        for i in range(1, len(filtered_window)):
            fx, _, fw = filtered_window[i]
            pm = mean
            w_sum += fw
            mean = pm + (fw / w_sum) * (fx - pm)
            var = var + fw * np.multiply(fx - pm, fx - mean)

        self.centroids[new_class][self.next_centroid_idx] = (mean, w_sum, np.array(var))
        self.buffers[new_class][self.next_centroid_idx] = filtered_window
        self.centroids_window_counts[new_class][self.next_centroid_idx] = self.centroids_window_counts[centroid_y][centroid_idx].copy() \
            if not split else Counter({new_class: self.centroids_window_counts[centroid_y][centroid_idx].get(new_class)})
        self.centroids_window_buffers[new_class][self.next_centroid_idx] = window_buffer.copy() if not split else filtered_window.copy()
        self.centroids_window_last_update[new_class][self.next_centroid_idx] = self.t
        self.next_centroid_idx += 1
        self.total_num_centroids += 1

    def __remove_centroid(self, centroid_y, centroid_idx):
        del self.centroids[centroid_y][centroid_idx]
        del self.buffers[centroid_y][centroid_idx]
        del self.centroids_window_counts[centroid_y][centroid_idx]
        del self.centroids_window_buffers[centroid_y][centroid_idx]
        del self.centroids_window_last_update[centroid_y][centroid_idx]
        self.total_num_centroids -= 1

    def __check_centroids(self):
        centroids = copy.deepcopy(self.centroids)

        for cls, class_centroids in centroids.items():
            for centroid_idx, centroid in class_centroids.items():
                if self.t - self.centroids_window_last_update[cls][centroid_idx] >= self.split_period:
                    if sum(self.centroids_window_counts[cls][centroid_idx].values()) < 0.4 * self.window_size:
                        self.__remove_centroid(cls, centroid_idx)
                    else:
                        centroid_split, sec_cls = self.__check_centroid_split(cls, centroid_idx)

                        if centroid_split:
                            self.splits_num += 1
                            self.__extract_new_centroid(cls, centroid_idx, cls, split=True)
                            self.__extract_new_centroid(cls, centroid_idx, sec_cls, split=True)
                            self.__remove_centroid(cls, centroid_idx)

                        self.centroids_window_last_update[cls][centroid_idx] = self.t

    def __check_centroid_split(self, centroid_y, centroid_idx):
        counts = self.centroids_window_counts[centroid_y][centroid_idx]
        if len(counts) < 2:
            return False, -1

        [(first_cls, first_cnt), (sec_cls, sec_cnt)] = counts.most_common(2)
        if centroid_y != first_cls or sec_cnt == 0:
            return False, -1

        return (first_cnt / sec_cnt) - 1.0 <= self.split_thresh, sec_cls

    def sample(self, x_batch, y_batch, weights, **kwargs):
        num_samples_per_instance = self.total_num_centroids
        num_samples = num_samples_per_instance * len(x_batch)

        input_shape = x_batch[0].shape
        sampled_x_batch = np.zeros((num_samples, *input_shape))
        sampled_y_batch = np.zeros(num_samples)
        sampled_weights = np.zeros(num_samples)

        cls_indices = collections.defaultdict(list)
        i = 0

        for _ in range(len(x_batch)):
            for class_idx, centroid_buffers in self.buffers.items():
                for buffer_idx, centroid_buffer in centroid_buffers.items():
                    if self.__try_sample(self.centroids_window_counts[class_idx][buffer_idx], class_idx):
                        (rx, ry, rw) = random.choice(centroid_buffer)
                        sampled_x_batch[i, :] = rx[:]
                        sampled_y_batch[i] = ry
                        sampled_weights[i] = rw
                        cls_indices[ry].append(i)

                        i += 1

        return self.__resample(sampled_x_batch[:i], sampled_y_batch[:i], sampled_weights[:i], cls_indices)

    @staticmethod
    def __try_sample(counts, cls):
        if len(counts) == 1 and list(counts.keys())[0] == cls:
            return True
        else:
            [(first_cls, first_cnt), (sec_cls, sec_cnt)] = counts.most_common(2)
            if first_cls == cls:
                r = math.tanh(4 * (first_cnt - sec_cnt) / (first_cnt + sec_cnt))
                return r > random.random()
        return False

    @staticmethod
    def __resample(x_batch, y_batch, weights, cls_indices):
        max_cnt = max([len(indices) for indices in cls_indices.values()])
        num_samples = len(cls_indices) * max_cnt

        input_shape = x_batch[0].shape
        resampled_x_batch = np.zeros((num_samples, *input_shape))
        resampled_y_batch = np.zeros(num_samples)
        resampled_weights = np.zeros(num_samples)

        i = 0
        for cls, indices in cls_indices.items():
            while len(indices) < max_cnt:
                indices.append(random.choice(indices))

            resampled_x_batch[i:i + max_cnt, :] = x_batch[indices, :]
            resampled_y_batch[i:i + max_cnt] = y_batch[indices]
            resampled_weights[i:i + max_cnt] = weights[indices]
            i += len(indices)

        return resampled_x_batch, resampled_y_batch, resampled_weights


class ExperienceReplay(ContinualLearner):

    def __init__(self, model: NeuralNet, replay_buffer: ReplayBuffer):
        super().__init__()
        self.model = model
        self.replay_buffer = replay_buffer

    def initialize(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', torch.ones(len(y_batch)))
        y_batch = y_batch.long()

        self.replay_buffer.add(x_batch, y_batch, weights)
        self.model.update(x_batch, y_batch, weights=weights)

    def predict(self, x_batch):
        return self.model.predict(x_batch)

    def predict_prob(self, x_batch):
        return self.model.predict_prob(x_batch)

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', torch.ones(len(y_batch)))
        y_batch = y_batch.long()

        sampled_x_batch, sampled_y_batch, sampled_weights = self.replay_buffer.sample(x_batch, y_batch, weights)
        sampled_batch = (Tensor(sampled_x_batch), Tensor(sampled_y_batch), Tensor(sampled_weights))
        ext_x_batch, ext_y_batch, ext_weights = self.extend((x_batch, y_batch, weights), sampled_batch)

        self.model.update(ext_x_batch, ext_y_batch, weights=ext_weights)
        self.replay_buffer.add(x_batch, y_batch, weights)

    @staticmethod
    def extend(input_batch, sampled_batch):
        sampled_x_batch, sampled_y_batch, sampled_weights = sampled_batch
        x_batch, y_batch, weights = input_batch

        ext_x_batch = torch.vstack((x_batch, sampled_x_batch))
        ext_y_batch = torch.hstack((y_batch, sampled_y_batch))
        ext_weights = torch.hstack((weights, sampled_weights))

        return ext_x_batch, ext_y_batch, ext_weights
