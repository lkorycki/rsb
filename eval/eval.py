import collections
import copy
import os

import torch
from typing import Callable
from abc import ABC, abstractmethod
from skmultiflow.drift_detection import ADWIN
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.tensorboard as tb
import tensorflow as tf

from core.clearn import ContinualLearner
from data.stream import Stream, InstanceStream, ClassStream
from eval.tf_writers import TBScalars, TBImages


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        pass


class InstanceStreamEvaluator(Evaluator):

    def __init__(self, batch_size: int, shuffle=False, init_skip_frac=0.05, numpy=False, logdir_root: str='runs'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.init_skip_frac = init_skip_frac
        self.numpy = numpy
        self.logdir_root = logdir_root

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator

        print('[1/3] Preparing data')
        instance_stream: InstanceStream = stream_creator()
        instance_stream_loader = DataLoader(instance_stream.get_data(), batch_size=self.batch_size, shuffle=self.shuffle)

        print('[2/3] Preparing model')
        model = model_creator()

        init_data = instance_stream.get_init_data()
        n = len(init_data)
        if n > 0:
            print(f'Initializing model with {n} instances')
            init_data_loader = DataLoader(init_data, batch_size=n, shuffle=self.shuffle)
            inputs_batch, labels_batch = next(iter(init_data_loader))
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
            model.initialize(inputs_batch, labels_batch)

        print('[3/3] Preparing metrics')
        per_class_acc = {}
        acc = ADWIN()
        correct = 0.0
        all = 0.0
        init_skip_num = self.init_skip_frac * len(instance_stream)

        logdir = f'{self.logdir_root}/{model_label}'
        tb_writer = tb.SummaryWriter(logdir)

        print('Evaluating...')
        i = 0
        for inputs_batch, labels_batch in tqdm(instance_stream_loader):
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()

            i += len(inputs_batch)
            preds = model.predict(inputs_batch)
            model.update(inputs_batch, labels_batch)

            results = [int(int(p) == int(y)) for p, y in zip(preds, labels_batch)]
            correct += sum(results)
            all += len(inputs_batch)

            for r, l in zip(results, labels_batch):
                acc.add_element(float(r))
                l = int(l)

                if l not in per_class_acc:
                    per_class_acc[l] = ADWIN()
                per_class_acc[l].add_element(float(r))

            if i > init_skip_num:
                tb_writer.add_scalar(f'ALL/{stream_label}', acc.estimation, i)

                for c, c_acc in per_class_acc.items():
                    tb_writer.add_scalar(f'{stream_label}/{stream_label}-C{c}', c_acc.estimation, i)


class ClassStreamEvaluator(Evaluator):

    def __init__(self, batch_size: int, shuffle: bool, num_epochs: int, num_workers: int, numpy=False, vis=True, logdir_root: str='runs'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.numpy = numpy
        self.vis = vis
        self.logdir_root = logdir_root

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator

        print('[1/3] Preparing data')
        class_stream: ClassStream = stream_creator()
        train_class_stream = class_stream.get_train_data()
        test_class_stream = iter(class_stream.get_test_data())

        print('[2/3] Preparing model')
        model = model_creator()

        init_class_concept_mapping, init_data = class_stream.get_init_data()
        n = len(init_data)
        if n > 0:
            print(f'Initializing model with {n} instances')
            init_data_loader = DataLoader(init_data, batch_size=n, num_workers=self.num_workers, shuffle=self.shuffle)
            inputs_batch, labels_batch = next(iter(init_data_loader))
            labels_batch = Tensor([init_class_concept_mapping[int(cls.item())] for cls in labels_batch])
            if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
            model.initialize(inputs_batch, labels_batch)

        print('[3/3] Preparing metrics')
        logdir = f'{self.logdir_root}/{model_label}'
        tb_file_writer = tf.summary.create_file_writer(logdir)
        tb_file_writer.set_as_default()
        classes_test_data = {}
        class_test_concept_mapping = {}
        results = collections.defaultdict(list)

        print('Evaluating...')
        for i, class_batch_data in enumerate(tqdm(train_class_stream)):
            (class_idx, class_batch_train_data, class_concept_mapping) = class_batch_data
            (test_class_idx, class_batch_test_data, test_class_concept_mapping) = next(test_class_stream)

            assert class_idx == test_class_idx and class_concept_mapping == test_class_concept_mapping
            class_test_concept_mapping.update(class_concept_mapping)

            classes_test_data[class_idx] = DataLoader(class_batch_test_data, batch_size=self.batch_size, num_workers=self.num_workers)  # todo: subclasses
            if self.vis: TBImages.write_test_data(class_batch_test_data, i, stream_label, class_stream.cls_names)

            for j in tqdm(range(self.num_epochs)):
                train_data_loader = DataLoader(class_batch_train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
                for inputs_batch, labels_batch in train_data_loader:
                    labels_batch = Tensor([class_concept_mapping[int(cls.item())] for cls in labels_batch])
                    if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
                    model.update(inputs_batch, labels_batch)

                tasks_acc, _ = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)
                TBScalars.write_epoch_result(tasks_acc, j, stream_label, i)

            tasks_acc, (task_targets, task_preds) = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)

            for k, task_acc in enumerate(tasks_acc): results[k + 1].append(task_acc)
            results[0].append(sum(tasks_acc) / len(tasks_acc))

            TBScalars.write_tasks_results(stream_label, tasks_acc, i)
            TBImages.write_confusion_matrices(task_targets, task_preds, i, stream_label)

        write_result_to_file(model_label, stream_label, results)


class OfflineClassStreamEvaluator(Evaluator):

    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, numpy=False, vis=True, logdir_root: str='runs',
                 model_path: str=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.numpy = numpy
        self.vis = vis
        self.logdir_root = logdir_root
        self.model_path = model_path

    def evaluate(self, model_creator: (str, Callable[[], ContinualLearner]), data_creator: (str, Callable[[], Stream])):
        model_label, model_creator = model_creator
        stream_label, stream_creator = data_creator
        model = None

        print('[1/2] Preparing data')
        class_stream: ClassStream = stream_creator()
        train_class_stream = class_stream.get_train_data()
        test_class_stream = iter(class_stream.get_test_data())

        print('[2/2] Preparing metrics')
        logdir = f'{self.logdir_root}/{model_label}'
        tb_file_writer = tf.summary.create_file_writer(logdir)
        tb_file_writer.set_as_default()
        all_train_data = None
        classes_test_data = {}
        class_test_concept_mapping = {}
        results = collections.defaultdict(list)

        print('Evaluating...')
        for i, class_batch_data in enumerate(tqdm(train_class_stream)):
            (class_idx, class_batch_train_data, class_concept_mapping) = class_batch_data
            (test_class_idx, class_batch_test_data, test_class_concept_mapping) = next(test_class_stream)

            assert class_idx == test_class_idx and class_concept_mapping == test_class_concept_mapping
            class_test_concept_mapping.update(class_concept_mapping)

            all_train_data = all_train_data + class_batch_train_data if all_train_data is not None else class_batch_train_data
            all_train_data_loader = DataLoader(all_train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                               shuffle=True)

            classes_test_data[class_idx] = DataLoader(class_batch_test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=True)
            if self.vis: TBImages.write_test_data(class_batch_test_data, i, stream_label, class_stream.cls_names)

            model = model_creator()

            for j in tqdm(range(self.num_epochs)):
                for inputs_batch, labels_batch in all_train_data_loader:
                    labels_batch = Tensor([class_test_concept_mapping[int(cls.item())] for cls in labels_batch])
                    if self.numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()
                    model.update(inputs_batch, labels_batch)

                if hasattr(model, 'scheduler') and model.scheduler is not None:
                    model.scheduler.step()

                tasks_acc, _ = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)
                TBScalars.write_epoch_result(tasks_acc, j, stream_label, i)

            tasks_acc, (task_targets, task_preds) = evaluate_tasks(model, classes_test_data, class_test_concept_mapping, self.numpy)

            for k, task_acc in enumerate(tasks_acc): results[k + 1].append(task_acc)
            results[0].append(sum(tasks_acc) / len(tasks_acc))

            TBScalars.write_tasks_results(stream_label, tasks_acc, i)
            TBImages.write_confusion_matrices(task_targets, task_preds, i, stream_label)

        write_result_to_file(model_label, stream_label, results)

        if self.model_path:
            print(f'Saving model: {self.model_path}')
            torch.save(model.get_net().state_dict(), self.model_path)


def evaluate_tasks(model: ContinualLearner, classes_test_data, class_test_concept_mapping, numpy):
    classes_acc, class_targets, class_preds = [], [], []

    for j, class_test_data in classes_test_data.items():
        correct, all = 0.0, 0.0

        for inputs_batch, labels_batch in class_test_data:
            labels_batch = Tensor([class_test_concept_mapping[int(cls.item())] for cls in labels_batch.long()])
            if numpy: inputs_batch, labels_batch = inputs_batch.numpy(), labels_batch.numpy()

            preds_batch = model.predict(inputs_batch)
            results = [p == y for p, y in zip(preds_batch, labels_batch)]
            correct += sum(results)
            all += len(inputs_batch)

            class_targets += list(labels_batch)
            class_preds += list(preds_batch)

        acc = correct / all
        classes_acc.append(acc)  # todo: add per subclass

    return classes_acc, (class_targets, class_preds)


def write_result_to_file(model_label, stream_label, results):
    os.makedirs('results', exist_ok=True)
    path = f'results/{model_label}#{stream_label}.csv'
    print('Writing results to file ', path)

    num_tasks = len(results[0])

    f = open(path, 'w')
    for task_id, values in results.items():
        ext = [0.0] * (num_tasks - len(values))
        values = ext + values
        values = [str(f.item() if torch.is_tensor(f) else str(f)) for f in values]
        vals = ','.join(values)
        f.write(f'{task_id},{vals}\n')
    f.close()
