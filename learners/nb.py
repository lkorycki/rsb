import numpy as np
from sklearn.naive_bayes import GaussianNB
from core.clearn import ContinualLearner
from utils.stat_utils import Statistics, GaussianEstimator
from utils.cls_utils import ClassificationUtils


class NaiveBayes(ContinualLearner):

    def __init__(self, use_prior=True, log_prob=True):
        super().__init__()
        self.use_prior = use_prior
        self.log_prob = log_prob
        self.stats = None
        self.init = False

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        if not self.init:
            return np.zeros((len(x_batch), 1))

        return np.array([self.prob(x) for x in x_batch], dtype=object)

    def prob(self, x):
        return ClassificationUtils.naive_bayes_log_prob(x, self.stats, self.use_prior) if self.log_prob else \
            ClassificationUtils.naive_bayes_prob(x, self.stats, self.use_prior)

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        if not self.init:
            self.stats = Statistics(np.arange(x_batch.shape[1]), max(np.unique(y_batch)) + 1, GaussianEstimator)
            self.init = True

        for x, y, w in zip(x_batch, y_batch, weights):
            self.stats.update(x, y, w)


class NaiveBayesScikit(ContinualLearner):

    def __init__(self):
        super().__init__()
        self.nb = GaussianNB()
        self.init = False

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        if not self.init:
            return np.zeros((len(x_batch), 1))

        return self.nb.predict_proba(x_batch)

    def update(self, x_batch, y_batch, **kwargs):
        y_batch = y_batch.astype(int) if isinstance(y_batch, np.ndarray) else y_batch.int()

        if not self.init:
            self.init = True

        self.nb.partial_fit(x_batch, y_batch, classes=np.arange(10))
