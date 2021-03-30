import math
from scipy.stats import entropy
import numpy as np

from core.clearn import ContinualLearner
from utils.cls_utils import ClassificationUtils as cu
from utils.stat_utils import Statistics, GaussianEstimator


class HoeffdingTree(ContinualLearner):
    def __init__(self, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, rnd=False, att_split_est=False,
                 num_atts=0, num_cls=0):
        super().__init__()
        self.root = TreeNode(split_step, split_wait, hb_delta, tie_thresh, rnd, None, att_split_est, num_atts, num_cls)

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        return np.array([self.find_leaf(x).predict_prob(x) for x in x_batch], dtype=object)

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        for x, y, w in zip(x_batch, y_batch, weights):
            leaf = self.find_leaf(x)
            leaf.update(x, y, w if w > 0.0 else 1.0)
            leaf.attempt_split()

    def find_leaf(self, x):
        node = self.root
        while not node.is_leaf:
            att, thresh = node.split
            node = node.left if x[att] <= thresh else node.right

        return node


class TreeNode:
    def __init__(self, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, rnd=False, stats=None, att_split_est=False,
                 num_atts=0, num_cls=0):
        self.is_leaf = True
        self.left = None
        self.right = None
        self.split = (None, None)
        self.split_step = split_step
        self.split_wait = split_wait  # make it adaptive?
        self.last_split_try = 0.0
        self.cnt = 0.0
        self.hb_delta = hb_delta
        self.tie_thresh = tie_thresh

        self.mc_correct = 0.0
        self.nb_correct = 0.0

        self.rnd = rnd
        self.att_split_est = att_split_est

        if num_atts > 0:
            self.init(num_atts, num_cls, stats)
        else:
            self.num_atts = 0
            self.num_cls = 0
            self.split_atts = []
            self.stat_atts = []
            self.att_map = None
            self.stats = None

    def init(self, num_atts, num_cls, stats):
        self.num_atts = num_atts
        self.num_cls = num_cls
        self.split_atts = np.arange(0, num_atts) if not self.rnd else np.random.permutation(num_atts)[:int(math.sqrt(num_atts)) + 1]
        self.stat_atts = self.split_atts if not self.att_split_est else np.arange(0, num_atts)

        if stats is not None:
            s = stats.get_stats()
            self.stats = Statistics(self.stat_atts, self.num_cls, GaussianEstimator, self.att_split_est, *s)
        else:
            self.stats = Statistics(self.stat_atts, self.num_cls, GaussianEstimator, self.att_split_est)

    def update(self, x, y: int, w: float):
        if not self.stats:
            self.init(len(x), y + 1, None)

        self.mc_correct += w * (np.argmax(self.stats.cls_counts) == y)
        self.nb_correct += w * (np.argmax(cu.naive_bayes_log_prob(x, self.stats)) == y)
        self.cnt += w
        self.stats.update(x, y, w)
        self.num_cls = self.stats.num_cls

    def predict_prob(self, x):
        if not self.stats:
            return np.zeros(1)
        else:
            return cu.majority_class_prob(self.stats.cls_counts, self.stats.get_cls_count()) \
                if self.mc_correct >= self.nb_correct else cu.naive_bayes_log_prob(x, self.stats)

    def attempt_split(self):
        if self.cnt > 0 and self.cnt - self.last_split_try >= self.split_wait:
            best_split = self.__find_best_att_split()
            self.__try_best_split(best_split)

    def __find_best_att_split(self):
        best_split = (float('inf'), None, None, [], [])  # entropy, att idx, thresh val, left prob, right prob

        for att_idx in self.split_atts:
            mi, mx = np.around(self.stats.get_att_extr(att_idx), decimals=10)
            step = (mx - mi) * self.split_step

            s = mi + step
            while s < mx:
                ent, l_prob, r_prob = Statistics.calc_split_weighted_entropy(self.stats, att_idx, s, self.num_cls)
                if ent < best_split[0]:
                    best_split = [ent, att_idx, s, l_prob, r_prob]

                s += step

        return best_split

    def __try_best_split(self, best_split):
        if best_split[1] is None: return
        n = self.stats.get_cls_count()
        curr_entropy = entropy(self.stats.cls_counts / n)
        hb = self.hb(math.log(self.num_cls), self.hb_delta, n)

        if curr_entropy - best_split[0] > hb or hb < self.tie_thresh:  # remove tie?
            _, att_idx, thresh, l_prob, r_prob = best_split
            self.split = (att_idx, thresh)

            l_stats, r_stats = self.stats.split(att_idx, thresh, l_prob, r_prob)

            self.left = TreeNode(self.split_step, self.split_wait, self.hb_delta, self.tie_thresh, self.rnd, l_stats,
                                 self.att_split_est, self.num_atts, self.num_cls)
            self.right = TreeNode(self.split_step, self.split_wait, self.hb_delta, self.tie_thresh, self.rnd, r_stats,
                                  self.att_split_est, self.num_atts, self.num_cls)

            self.is_leaf = False
            self.stats = None

        self.last_split_try = self.cnt

    @staticmethod
    def hb(r, delta, n):
        return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))
