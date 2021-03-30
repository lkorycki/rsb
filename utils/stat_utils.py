from abc import ABC, abstractmethod
from typing import Callable
import math
import numpy as np
from scipy.stats import entropy
import copy

from utils.calc_utils import CalculationsUtils as cu
from utils.coll_utils import CollectionUtils as clu

SQ2 = math.sqrt(2.0)
NC = math.sqrt(2.0 * math.pi)


class ValueEstimator(ABC):

    @abstractmethod
    def update(self, v: float, w: float):
        pass

    @abstractmethod
    def get_count(self):
        pass

    @abstractmethod
    def get_mean(self):
        pass

    @abstractmethod
    def get_var(self):
        pass

    @abstractmethod
    def get_std(self):
        pass

    @abstractmethod
    def copy(self):
        pass


class ForgettingEstimator(ValueEstimator):
    def __init__(self, decay: float, count: float=0.0, linear_sum: float=0.0, squared_sum: float=0.0, timestamp: int=0):
        self.decay = decay
        self.count = count
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.timestamp = timestamp

    def update(self, v: float, t: int):
        d = self.decay ** (t - self.timestamp)
        self.count = d * self.count + 1
        self.linear_sum = d * self.linear_sum + v
        self.squared_sum = d * self.squared_sum + v**2

        self.timestamp = t

    def get_count(self):
        return self.count

    def get_mean(self):
        if self.count == 0.0:
            return float('NaN')
        print(self.linear_sum, self.count)
        return self.linear_sum / self.count

    def get_var(self):
        if self.count == 0.0:
            return float('NaN')
        return max(self.squared_sum / self.count - (self.linear_sum / self.count) ** 2, 0.0)

    def get_std(self):
        if self.count == 0.0:
            return float('NaN')
        return math.sqrt(self.get_var())

    def copy(self):
        return ForgettingEstimator(self.decay, self.count, self.linear_sum, self.squared_sum, self.timestamp)


class DistributionEstimator(ValueEstimator):

    @abstractmethod
    def get_cdf(self, x: float):
        pass

    @abstractmethod
    def get_pdf(self, x: float):
        pass

    @abstractmethod
    def split(self, **kwargs):
        pass


class GaussianEstimator(DistributionEstimator):
    def __init__(self, count: float=0.0, mean: float=0.0, var: float=0.0, std: float=0.0):
        self.count = count
        self.mean = mean
        self.var = var
        self.std = std

    def update(self, v: float, w: float):
        pm = self.mean
        self.count += w
        self.mean = pm + (w / self.count) * (v - pm)
        self.var = self.var + w * (v - pm) * (v - self.mean)
        self.std = math.sqrt(self.var / self.count)

    def get_count(self):
        return self.count

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def get_std(self):
        return self.std

    def get_cdf(self, x: float):
        z = (x - self.mean) / self.std
        return (1.0 + math.erf(z / SQ2)) / 2.0

    def get_pdf(self, x: float):
        if self.std == 0:
            return 1.0 if x == self.mean else 0.0

        z = (x - self.mean) / self.std
        return math.exp(-0.5 * z * z) / (NC * self.std)

    def split(self, **kwargs):
        return GaussianEstimator(kwargs.get('lc'), self.mean, self.var, self.std), \
               GaussianEstimator(kwargs.get('rc'), self.mean, self.var, self.std)

    def copy(self):
        return GaussianEstimator(self.count, self.mean, self.var, self.std)


class Statistics:
    def __init__(self, atts: list, num_cls: int, estimator_creator: Callable[[], DistributionEstimator], att_split_est=False,
                 cls_counts=None, att_stats=None, att_extr_stats=None):
        self.atts = atts
        self.att_map = {att_idx: i for i, att_idx in enumerate(self.atts)}
        self.num_cls = num_cls
        self.estimator_creator = estimator_creator
        self.att_split_est = att_split_est

        self.att_stats = att_stats if att_stats is not None else [[self.estimator_creator() for _ in range(num_cls)] for _ in range(len(self.atts))]
        self.att_extr_stats = att_extr_stats if att_extr_stats is not None else np.array([[float('inf'), float('-inf')] for _ in range(len(self.atts))])
        self.cls_counts = cls_counts if cls_counts is not None else np.zeros(num_cls)
        self.all_count = self.cls_counts.sum()

    def update(self, x, y: int, w: float):
        self.cls_counts = clu.ensure_arr_size(self.cls_counts, y)
        self.att_stats = clu.ensure_list2d_size(self.att_stats, y, self.estimator_creator)
        self.num_cls = max(self.num_cls, y + 1)

        self.all_count += w
        self.cls_counts[y] += w

        for i, att_idx in enumerate(self.atts):
            self.att_stats[i][y].update(x[att_idx], w)
            self.att_extr_stats[i][0] = min(self.att_extr_stats[i][0], x[att_idx])
            self.att_extr_stats[i][1] = max(self.att_extr_stats[i][1], x[att_idx])

    def get_estimator(self, att_idx: int, cls_idx: int):
        return self.att_stats[self.att_map[att_idx]][cls_idx]

    def get_cls_count(self, cls_idx: int=-1):
        return self.all_count if cls_idx < 0 else self.cls_counts[cls_idx]

    def get_att_extr(self, att_idx: int):
        return self.att_extr_stats[self.att_map[att_idx]]

    def get_stats(self):
        return self.cls_counts, self.att_stats, self.att_extr_stats

    def split(self, split_att_idx: int, s: float, l_prob, r_prob):
        l_cls_counts, r_cls_counts = self.all_count * l_prob, self.all_count * r_prob
        l_att_stats, r_att_stats = None, None
        l_att_extr, r_att_extr = None, None

        if self.att_split_est:
            l_att_stats, r_att_stats = [], []
            for i, att_idx in enumerate(self.atts):
                lstats, rstats = [], []

                for cls_idx in range(self.num_cls):
                    lest, rest = self.get_estimator(i, cls_idx).split(lc=l_cls_counts[cls_idx], rc=r_cls_counts[cls_idx])
                    lstats.append(lest)
                    rstats.append(rest)

                l_att_stats.append(lstats)
                r_att_stats.append(rstats)

            l_att_extr, r_att_extr = copy.deepcopy(self.att_extr_stats), copy.deepcopy(self.att_extr_stats)
            l_att_extr[split_att_idx][1] = s
            r_att_extr[split_att_idx][0] = s

        return Statistics(self.atts, self.num_cls, self.estimator_creator, self.att_split_est, l_cls_counts, l_att_stats, l_att_extr), \
               Statistics(self.atts, self.num_cls, self.estimator_creator, self.att_split_est, r_cls_counts, r_att_stats, r_att_extr)

    @staticmethod
    def calc_split_weighted_entropy(stats, att_idx: int, s: float, num_cls: int):
        l_prob, r_prob = np.zeros(num_cls), np.zeros(num_cls)
        lp_sum, rp_sum = 0.0, 0.0

        for cls_idx in range(num_cls):
            est = stats.get_estimator(att_idx, cls_idx)

            if est.get_count() == 0:
                continue
            elif est.get_var() == 0:
                lp = 1.0 if s >= est.get_mean() else 0.0
            else:
                lp = est.get_cdf(s)

            lp_sum += lp
            rp = 1.0 - lp
            rp_sum += rp

            cp = stats.get_cls_count(cls_idx) / stats.get_cls_count()
            l_prob[cls_idx] = lp * cp
            r_prob[cls_idx] = rp * cp

        wl, wr = l_prob.sum(), r_prob.sum()
        l_ent = entropy(cu.div_tensor(l_prob, lp_sum)) if wl > 0.0 else 0.0
        r_ent = entropy(cu.div_tensor(r_prob, rp_sum)) if wr > 0.0 else 0.0
        ent = wl * l_ent + wr * r_ent

        return ent, l_prob, r_prob
