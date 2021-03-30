from core.clearn import ContinualLearner
from data.stream import Stream
from eval.eval import Evaluator

import itertools
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from torch.utils.data import Dataset


class Experiment(ABC):

    def __init__(self):
        self.algorithms: Dict[str, Callable[[], ContinualLearner]] = {}
        self.streams: Dict[str, Callable[[], Stream]] = {}
        self.evaluators: Dict[str, Callable[[], Evaluator]] = {}

    def run(self, algorithms: List[str] = None, streams: List[str] = None, evaluators: List[str] = None):
        self.prepare()
        algorithms = self.algorithms.keys() if not algorithms else algorithms
        streams = self.streams.keys() if not streams else streams
        evaluators = self.evaluators.keys() if not evaluators else evaluators

        for a, s, e, in itertools.product(algorithms, streams, evaluators):
            print(f'Running for: {a}, {s}, {e}')
            self.evaluators[e]().evaluate((a, self.algorithms[a]), (s, self.streams[s]))

    def add_algorithm_creator(self, label: str, algorithm: Callable[[], ContinualLearner]):
        self.algorithms[label] = algorithm

    def add_data_creator(self, label: str, stream: Callable[[], Stream]):
        self.streams[label] = stream

    def add_evaluator_creator(self, label: str, evaluator: Callable[[], Evaluator]):
        self.evaluators[label] = evaluator

    @abstractmethod
    def prepare(self):
        pass
