import math

from skmultiflow.core import ClassifierMixin
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np

from core.clearn import ContinualLearner
from utils.calc_utils import CalculationsUtils
from utils.stat_utils import Statistics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ClassificationUtils:

    @staticmethod
    def majority_class_prob(counts, all_count):
        return np.zeros(len(counts)) if all_count == 0 else counts / all_count

    @staticmethod
    def naive_bayes_prob(x, stats: Statistics, use_prior=True):  # rewrite to C
        probs = np.zeros(stats.num_cls, dtype=np.double)
        if stats.get_cls_count() == 0: return probs
        p = 0.0

        for cls_idx in range(stats.num_cls):
            prob = stats.get_cls_count(cls_idx) / stats.get_cls_count() if use_prior else 1.0

            for att_idx in stats.atts:
                pr = stats.get_estimator(att_idx, cls_idx).get_pdf(x[att_idx])
                prob *= (pr if not np.isnan(pr) else 1.0)

            p += prob
            probs[cls_idx] = prob

        return probs if p == 0.0 else probs / p

    @staticmethod
    def naive_bayes_log_prob(x, stats: Statistics, use_prior=True):
        log_probs = np.zeros(stats.num_cls, dtype=np.double)
        if stats.get_cls_count() == 0: return log_probs
        lps, lp_max = [], float('-inf')

        for cls_idx in range(stats.num_cls):
            log_prob = CalculationsUtils.log(stats.get_cls_count(cls_idx) / stats.get_cls_count()) if use_prior else 0.0

            for att_idx in stats.atts:
                pr = stats.get_estimator(att_idx, cls_idx).get_pdf(x[att_idx])
                log_prob += (CalculationsUtils.log(pr) if not np.isnan(pr) else 0.0)

            lps.append(log_prob)
            lp_max = max(lp_max, log_prob)
            log_probs[cls_idx] = log_prob

        # https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes/253319#253319
        lps_sum = 0.0
        zero_prob_indices = []
        for i, lp in enumerate(lps):
            if lp != float('-inf'):
                lps_sum += math.exp(lp - lp_max)
            else:
                zero_prob_indices.append(i)

        if not (np.isfinite(lps_sum) and np.isfinite(lp_max)):
            return np.zeros(stats.num_cls)

        lps_sum = math.log(lps_sum)
        lps_sum += lp_max

        probs = np.exp(log_probs - lps_sum)
        probs[zero_prob_indices] = 0.0

        return probs


class StreamingWrapper(ContinualLearner):

    def __init__(self, classifier: ClassifierMixin):
        super().__init__()
        self.classifier = classifier

    def predict(self, x_batch):
        return self.classifier.predict(x_batch)

    def predict_prob(self, x_batch):
        return self.classifier.predict_proba(x_batch)

    def update(self, x_batch, y_batch, **kwargs):
        self.classifier.partial_fit(x_batch, y_batch)


class StreamingHybrid(ContinualLearner):

    def __init__(self, feat_extr: nn.Module, classifier: ContinualLearner, numpy=True, device='cpu'):
        super().__init__()
        self.feat_extr = feat_extr.to(device)
        self.classifier = classifier
        self.numpy = numpy
        self.device = device

    def initialize(self, x_batch: Tensor, y_batch: Tensor, **kwargs):
        feat_outputs = self.feat_extr(x_batch.to(self.device)).cpu()
        if self.numpy: feat_outputs, y_batch, = feat_outputs.numpy(), y_batch.numpy()
        self.classifier.initialize(feat_outputs, y_batch, **kwargs)

    def predict(self, x_batch: Tensor):
        feat_outputs = self.feat_extr(x_batch.to(self.device)).cpu()
        if self.numpy: feat_outputs = feat_outputs.numpy()
        return np.array([np.argmax(ya) for ya in self.classifier.predict_prob(feat_outputs)])

    def predict_prob(self, x_batch: Tensor):
        feat_outputs = self.feat_extr(x_batch.to(self.device)).cpu()
        if self.numpy: feat_outputs = feat_outputs.numpy()
        probs = self.classifier.predict_prob(feat_outputs)
        return probs

    def update(self, x_batch: Tensor, y_batch: Tensor, **kwargs):
        feat_outputs = self.feat_extr(x_batch.to(self.device)).cpu()
        if self.numpy: feat_outputs, y_batch = feat_outputs.numpy(), y_batch.numpy()
        self.classifier.update(feat_outputs, y_batch, **kwargs)


class StreamingHybridWrapper(ContinualLearner):

    def __init__(self, feat_extr: nn.Module, classifier: ClassifierMixin):
        super().__init__()
        self.feat_extr = feat_extr
        self.classifier = classifier

    def predict(self, x_batch: Tensor):
        feat_outputs = self.feat_extr(x_batch.to(device)).cpu()
        outputs = Tensor(self.classifier.predict_proba(feat_outputs.numpy()))
        return torch.max(outputs, 1)[1]

    def predict_prob(self, x_batch: Tensor):
        feat_outputs = self.feat_extr(x_batch.to(device)).cpu()
        return Tensor(self.classifier.predict_proba(feat_outputs.numpy()))

    def update(self, x_batch: Tensor, y_batch: Tensor, **kwargs):
        feat_outputs = self.feat_extr(x_batch.to(device)).cpu()
        self.classifier.partial_fit(feat_outputs.numpy(), y_batch.numpy())


class RandomClassifier(ContinualLearner):

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def predict(self, x_batch):
        return torch.randint(0, self.num_classes, (len(x_batch),))

    def predict_prob(self, x_batch):
        return torch.rand((len(x_batch), self.num_classes))

    def update(self, x_batch, y_batch, **kwargs):
        pass
