import math
import random
from collections import Counter

from skmultiflow.drift_detection import ADWIN
import numpy as np
import ray

from core.clearn import ContinualLearner
from learners.ht import HoeffdingTree
from learners.nb import NaiveBayes
from utils.calc_utils import CalculationsUtils
from utils.coll_utils import CollectionUtils


class IncrementalRandomForest(ContinualLearner):

    def __init__(self, size: int, lambda_val=5.0, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05,
                 att_split_est=False, num_atts=0, num_cls=0, num_workers=1):
        super().__init__()
        self.size = size
        self.lambda_val = lambda_val
        self.tree_groups = []
        self.num_par_groups = num_workers
        self.par = self.num_par_groups > 1
        self.init_tree_groups([0, split_step, split_wait, hb_delta, tie_thresh, att_split_est, num_atts, num_cls])

    def init_tree_groups(self, tree_params: list):
        trees_per_group, r = math.ceil(self.size / self.num_par_groups), self.size
        while r > 0:
            tree_params[0] = min(r, trees_per_group)
            self.tree_groups.append(RemoteTreeGroupWrapper(*tree_params) if self.par else TreeGroup(*tree_params))
            r -= trees_per_group

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        weights = self.fetch([tg.get_weights() for tg in self.tree_groups])
        ws = sum([sum(w) for w in weights])

        probs = self.fetch([tg.predict_prob(x_batch) for tg in self.tree_groups])
        trees_batch_probs = CollectionUtils.flatten_list(probs)
        probs_sum = [CalculationsUtils.sum_arrays([tree_probs[i] for tree_probs in trees_batch_probs]) for i in range(len(x_batch))]

        return np.array(probs_sum, dtype=object) / ws

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))

        for tree_group in self.tree_groups:
            tree_group.update_trees(x_batch, y_batch, self.lambda_val, weights)

    def get_tree_group(self, idx):
        return self.tree_groups[idx].get_trees()

    def fetch(self, obj):
        return obj if not self.par else ray.get(obj)


class TreeGroup:

    def __init__(self, size: int, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05,
                 att_split_est=False, num_atts=0, num_cls=0):
        self.trees = [
            ForestHoeffdingTree(split_step, split_wait, hb_delta, tie_thresh, att_split_est, num_atts, num_cls)
            for _ in range(size)
        ]

    def get_weights(self):
        return [tree.get_weight() for tree in self.trees]

    def predict_prob(self, x_batch):
        return [tree.predict_prob(x_batch) for tree in self.trees]

    def update_trees(self, x_batch, y_batch, lambda_val, weights):
        for tree in self.trees:
            k = np.random.poisson(lambda_val, len(x_batch))
            tree.update(x_batch, y_batch, weights=np.multiply(weights, k))

    def get_trees(self):
        return self.trees


class ForestHoeffdingTree(HoeffdingTree):

    def __init__(self, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, att_split_est=False, num_atts=0, num_cls=0):
        super().__init__(split_step, split_wait, hb_delta, tie_thresh, True, att_split_est, num_atts, num_cls)
        self.quality = ADWIN()

    def update(self, x_batch, y_batch, **kwargs):
        preds = super().predict(x_batch)
        for p, y in zip(preds, y_batch): self.quality.add_element(int(int(p) == int(y)))
        super().update(x_batch, y_batch, **kwargs)

    def predict_prob(self, x_batch):
        return self.get_weight() * np.array([self.find_leaf(x).predict_prob(x) for x in x_batch], dtype=object)

    def get_weight(self):
        return self.quality.estimation if self.quality.total > 0 else 1.0


@ray.remote
class RemoteTreeGroup(TreeGroup):

    def __init__(self, size: int, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05,
                 att_split_est=False, num_atts=0, num_cls=0):
        super().__init__(size, split_step, split_wait, hb_delta, tie_thresh, att_split_est, num_atts, num_cls)


class RemoteTreeGroupWrapper:

    def __init__(self, *remote_tree_group_args):
        self.remote_tree_group = RemoteTreeGroup.remote(*remote_tree_group_args)

    def get_weights(self):
        return self.remote_tree_group.get_weights.remote()

    def predict_prob(self, x_batch):
        return self.remote_tree_group.predict_prob.remote(x_batch)

    def update_trees(self, x_batch, y_batch, lambda_val, weights):
        self.remote_tree_group.update_trees.remote(x_batch, y_batch, lambda_val, weights)

    def get_trees(self):
        return self.remote_tree_group.get_trees.remote()


class IncrementalSubforests(ContinualLearner):

    def __init__(self, subforest_size: int, replay_max_size=1000, lambda_val=5.0, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05,
                 att_split_est=False, uncertainty_alpha=0.0, num_atts=0):
        super().__init__()
        self.subforest_size = subforest_size
        self.replay_max_size = replay_max_size
        self.unc_alpha = uncertainty_alpha

        self.subforests = {}
        self.replay_buffer = []  # todo: class BalancedReplayBuffer()
        self.buffer_counts = Counter()
        self.cls_indices = {}

        self.subforest_params = {
            'lambda_val': lambda_val,
            'split_step': split_step,
            'split_wait': split_wait,
            'hb_delta': hb_delta,
            'tie_thresh': tie_thresh,
            'att_split_est': att_split_est,
            'num_atts': num_atts,
            'num_cls': 2
        }

    def initialize(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        for x, y, w in zip(x_batch, y_batch, weights):
            if len(self.replay_buffer) < self.replay_max_size:
                self.replay_buffer.append((x, y, w))
                self.buffer_counts[y] += 1

        self.update(x_batch, y_batch, init=True)

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        max_cls_idx = max(self.subforests.keys()) if len(self.subforests) > 0 else 0
        probs = np.zeros((len(x_batch), max_cls_idx + 1))

        for cls, subforest in self.subforests.items():
            unc_scaling = (1.0 - self.unc_alpha + math.tanh(2 * cls / len(self.subforests)) * self.unc_alpha)
            probs[:, cls] = unc_scaling * subforest.predict_prob(x_batch)[:, 1]  # log-prob

        return probs

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        cls_indices = {}
        for i, y in enumerate(y_batch):
            if y in cls_indices:
                cls_indices[y].append(i)
            else:
                cls_indices[y] = [i]

        for cls, indices in cls_indices.items():
            if cls not in self.subforests:
                self.subforests[cls] = IncrementalRandomForest(self.subforest_size, **self.subforest_params)  # todo: num_cpus?

            filtered_replay_buffer = list(filter(lambda r: r[1] != cls, self.replay_buffer))

            for idx in indices:
                self.subforests[cls].update([x_batch[idx]], np.array([1]), weights=[weights[idx]])

                rx, ry, rw = random.choice(filtered_replay_buffer)
                self.subforests[cls].update([rx], np.array([0]), weights=[rw])

            if kwargs.get('init', False): continue
            available_indices = list(range(len(self.replay_buffer)))

            for idx in indices:
                if len(self.replay_buffer) == self.replay_max_size:
                    r = random.choice(available_indices)
                    cc, cr = self.buffer_counts[cls], self.buffer_counts[self.replay_buffer[r][1]]

                    if cc < cr:  # compare with a simpler method, show class balance/diversity
                        available_indices.remove(r)  # faster?
                        self.buffer_counts[self.replay_buffer[r][1]] -= 1

                        self.replay_buffer[r] = (x_batch[idx], y_batch[idx], weights[idx])
                        self.buffer_counts[cls] += 1
                else:
                    self.replay_buffer.append((x_batch[idx], y_batch[idx], weights[idx]))
                    self.buffer_counts[cls] += 1


class IncrementalBayesianEnsemble(ContinualLearner):

    def __init__(self, use_prior):
        super().__init__()
        self.use_prior = use_prior
        self.ensemble = {}

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        max_cls_idx = int(max(self.ensemble.keys())) if len(self.ensemble) > 0 else 0
        probs = np.zeros((len(x_batch), max_cls_idx + 1))

        for cls, nb in self.ensemble.items():
            probs[:, int(cls)] = nb.predict_prob(x_batch)[:, 1]

        return probs

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        cls_indices = {}
        for i, y in enumerate(y_batch):
            if y in cls_indices:
                cls_indices[y].append(i)
            else:
                cls_indices[y] = [i]

        for cls, indices in cls_indices.items():
            if cls not in self.ensemble:
                self.ensemble[cls] = NaiveBayes(use_prior=self.use_prior)

            self.ensemble[cls].update(x_batch[indices], np.ones(len(indices)), weights=weights[indices])

            for cls_idx, nb in self.ensemble.items():  # todo: parallel
                if cls_idx != cls:
                    nb.update(x_batch[indices], np.zeros(len(indices)), weights=weights[indices])

        print(self.ensemble)
