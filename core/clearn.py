from abc import ABC, abstractmethod


class ContinualLearner(ABC):

    def __init__(self):
        pass

    def initialize(self, x_batch, y_batch, **kwargs):
        self.update(x_batch, y_batch, **kwargs)

    @abstractmethod
    def predict(self, x_batch):
        pass

    @abstractmethod
    def predict_prob(self, x_batch):
        pass

    @abstractmethod
    def update(self, x_batch, y_batch, **kwargs):
        pass
