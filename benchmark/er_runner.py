import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam

import data.data_collection as data_col
from data.stream import ClassStream
from eval.eval import ClassStreamEvaluator
from eval.experiment import Experiment
from learners.nnet import NeuralNet
from learners.er import ExperienceReplay, ClassBuffer, SubspaceBuffer, ReactiveSubspaceBuffer

num_cores = multiprocessing.cpu_count()


class ExperimentExperienceReplay(Experiment):

    def prepare(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.multiprocessing.set_sharing_strategy('file_system')
        logdir_root = 'runs/er'

        def mlp_creator(input_size=128, output_size=10):
            net = NeuralNet.create_nn((input_size, 512, 256, 128, output_size), 0.5)
            optimizer = Adam(net.parameters())
            return NeuralNet(net, CrossEntropyLoss(), optimizer, None, device=device)

        self.add_algorithm_creator('NN', lambda: mlp_creator())
        self.add_algorithm_creator('ER-CB-2000-p0', lambda: ExperienceReplay(mlp_creator(), ClassBuffer(buffer_max_size=2000, replace_prob=0.0)))
        self.add_algorithm_creator('ER-CB-2000-p1', lambda: ExperienceReplay(mlp_creator(), ClassBuffer(buffer_max_size=2000, replace_prob=1.0)))
        self.add_algorithm_creator('ER-SB10x100', lambda: ExperienceReplay(mlp_creator(), SubspaceBuffer(max_centroids=10, max_instances=100)))
        self.add_algorithm_creator('ER-SB20x100', lambda: ExperienceReplay(mlp_creator(), SubspaceBuffer(max_centroids=20, max_instances=100)))
        self.add_algorithm_creator('ER-RSB10x100', lambda: ExperienceReplay(mlp_creator(),
                                                                            ReactiveSubspaceBuffer(max_centroids=10, max_instances=100,
                                                                                                   window_size=100, split=True)))
        self.add_algorithm_creator('ER-RSB20x100', lambda: ExperienceReplay(mlp_creator(),
                                                                            ReactiveSubspaceBuffer(max_centroids=20, max_instances=100,
                                                                                                   window_size=100, split=True)))

        streams = [
            ('MNIST-REC-TENSOR', 'MNIST-TRAIN-TENSOR', 'MNIST-TEST-TENSOR'),
            ('FASHION-REC-TENSOR', 'FASHION-TRAIN-TENSOR', 'FASHION-TEST-TENSOR'),
            ('SVHN-REC-TENSOR', 'SVHN-TRAIN-TENSOR', 'SVHN-TEST-TENSOR'),
            ('IMAGENET10-REC-TENSOR', 'IMAGENET10-TRAIN-TENSOR', 'IMAGENET10-TEST-TENSOR'),
            ('CIFAR10-REC-TENSOR', 'CIFAR10-TRAIN-TENSOR', 'CIFAR10-TEST-TENSOR')
        ]

        rec_seq_stat_10 = [(0, [0], {0: 1}), (1, [1], {1: 0}), (2, [2], {2: 1}), (3, [3], {3: 0}), (4, [4], {4: 1}),
                           (5, [5], {5: 0}), (6, [6], {6: 1}), (7, [7], {7: 0}), (8, [8], {8: 1}), (9, [9], {9: 0})]

        def data_creator(train, test, seq):
            return lambda: ClassStream(data_col.get(train), data_col.get(test), init_data={0: 0.1, 1: 0.1}, class_batch_seq=seq)

        for name, train, test in streams:
            self.add_data_creator(f'{name}-S1', data_creator(train, test, rec_seq_stat_10))

        rec_seq_drift_2_20 = [(0, [0], {0: 1}), (1, [1], {1: 0}), (2, [2], {2: 1}), (3, [3], {3: 0}),
                              (0, [0], {0: 0}), (1, [1], {1: 1}),
                              (4, [4], {4: 1}), (5, [5], {5: 0}), (6, [6], {6: 1}),
                              (2, [2], {2: 0}), (3, [3], {3: 1}),
                              (7, [7], {7: 0}), (8, [8], {8: 1}), (9, [9], {9: 0}),
                              (4, [4], {4: 0}), (5, [5], {5: 1}),
                              (0, [0], {0: 0}), (1, [1], {1: 1}), (2, [2], {2: 0}),
                              (6, [6], {6: 0}), (7, [7], {7: 1}),
                              (3, [3], {3: 1}), (4, [4], {4: 0}), (5, [5], {5: 1}),
                              (8, [8], {8: 0}), (9, [9], {9: 1}),
                              (0, [0], {0: 0}), (1, [1], {1: 1}), (2, [2], {2: 0}), (3, [3], {3: 1})]

        for name, train, test in streams:
            self.add_data_creator(f'{name}-D1', data_creator(train, test, rec_seq_drift_2_20))

        self.add_evaluator_creator('IncEval-ep10', lambda: ClassStreamEvaluator(batch_size=256, shuffle=True, num_epochs=10,
                                                                           num_workers=8, logdir_root=logdir_root,
                                                                           numpy=False, vis=False))

        self.add_evaluator_creator('IncEval-ep5', lambda: ClassStreamEvaluator(batch_size=256, shuffle=True, num_epochs=5,
                                                                           num_workers=8, logdir_root=logdir_root,
                                                                           numpy=False, vis=False))


def run():

    # Stationary
    ExperimentExperienceReplay().run(algorithms=['NN', 'ER-CB0', 'ER-CB1'],
                                     streams=['CIFAR10-REC-TENSOR-S1', 'SVHN-REC-TENSOR-S1',
                                              'FASHION-REC-TENSOR-S1', 'MNIST-REC-TENSOR-S1', 'IMAGENET10-REC-TENSOR-S1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB10x100', 'ER-RSB10x100'],
                                     streams=['CIFAR10-REC-TENSOR-S1', 'SVHN-REC-TENSOR-S1', 'MNIST-REC-TENSOR-S1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB20x100', 'ER-RSB20x100'],
                                     streams=['FASHION-REC-TENSOR-S1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB10x100', 'ER-RSB10x100'],
                                     streams=['IMAGENET10-REC-TENSOR-S1'],
                                     evaluators=['IncEval-ep5'])

    # Drifting
    ExperimentExperienceReplay().run(algorithms=['NN', 'ER-CB0', 'ER-CB1'],
                                     streams=['CIFAR10-REC-TENSOR-D1', 'SVHN-REC-TENSOR-D1',
                                              'FASHION-REC-TENSOR-D1', 'MNIST-REC-TENSOR-D1', 'IMAGENET10-REC-TENSOR-D1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB10x100', 'ER-RSB10x100'],
                                     streams=['CIFAR10-REC-TENSOR-D1', 'SVHN-REC-TENSOR-D1', 'MNIST-REC-TENSOR-D1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB20x100', 'ER-RSB20x100'],
                                     streams=['FASHION-REC-TENSOR-D1'],
                                     evaluators=['IncEval-ep10'])

    ExperimentExperienceReplay().run(algorithms=['ER-SB10x100', 'ER-RSB10x100'],
                                     streams=['IMAGENET10-REC-TENSOR-D1'],
                                     evaluators=['IncEval-ep5'])
