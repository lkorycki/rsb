import unittest

from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.stream import InstanceStream, ClassStream


class StreamTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_instance_stream(self):
        inputs, labels = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), [2, 3, 0, 1, 2]

        batch_size = 2
        instance_stream = InstanceStream(DummyDataset(inputs, labels), init_frac=0.0)
        self.__test_instance_stream(instance_stream, inputs, labels, batch_size)

        batch_size = 1
        instance_stream = InstanceStream(DummyDataset(inputs, labels))
        self.__test_instance_stream(instance_stream, inputs, labels, batch_size)

        batch_size = 2
        order = [3, 4, 0, 2, 1]
        instance_stream = InstanceStream(DummyDataset(inputs, labels), order=order, init_frac=0.0)
        inputs, labels = np.array([[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0]]), [1, 2, 2, 0, 3]
        self.__test_instance_stream(instance_stream, inputs, labels, batch_size)

        inputs, labels = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), [2, 3, 0, 1, 2]
        batch_size = 2
        instance_stream = InstanceStream(DummyDataset(inputs, labels), init_frac=0.4)
        self.__test_instance_stream(instance_stream, inputs[2:], labels[2:], batch_size)

        init_data = instance_stream.get_init_data()
        init_data_loader = DataLoader(init_data, batch_size=len(init_data))
        inputs, labels = next(iter(init_data_loader))
        np.testing.assert_equal(inputs, np.array([[0.0, 0.0], [1.0, 1.0]]))
        np.testing.assert_equal(labels, np.array([2, 3]))

    @staticmethod
    def __test_instance_stream(instance_stream, inputs, labels, batch_size):
        loader = DataLoader(instance_stream.get_data(), batch_size=batch_size, shuffle=False)
        i = 0

        for inputs_batch, labels_batch in loader:
            np.testing.assert_equal(inputs_batch.numpy(), inputs[i * batch_size:i * batch_size + batch_size])
            np.testing.assert_equal(labels_batch.numpy(), labels[i * batch_size:i * batch_size + batch_size])
            i += 1

    def test_class_stream(self):
        inputs, labels = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]), [2, 3, 0, 1, 2, 2]
        train_dataset = test_dataset = DummyDataset(inputs, labels)

        class_stream = ClassStream(train_dataset, test_dataset, class_size=1)
        class_seq = [
            (0, [0], {0: 0}),
            (1, [1], {1: 1}),
            (2, [2], {2: 2}),
            (3, [3], {3: 3})
        ]
        batches_inputs = [
            [[2.0, 2.0]],
            [[3.0, 3.0]],
            [[0.0, 0.0], [4.0, 4.0], [5.0, 5.0]],
            [[1.0, 1.0]]
        ]

        self.__test_class_stream(class_stream, class_seq, batches_inputs)
        for i, (_, _, cm) in enumerate(class_stream.get_train_data()): self.assertDictEqual(cm, class_seq[i][-1])

        class_stream = ClassStream(train_dataset, test_dataset, class_size=2)
        class_seq = [
            (0, [0, 1], {0: 0, 1: 0}),
            (1, [2, 3], {2: 1, 3: 1})
        ]
        batches_inputs = [
            [[2.0, 2.0], [3.0, 3.0]],
            [[0.0, 0.0], [4.0, 4.0], [5.0, 5.0], [1.0, 1.0]]
        ]

        self.__test_class_stream(class_stream, class_seq, batches_inputs)
        for i, (_, _, cm) in enumerate(class_stream.get_train_data()): self.assertDictEqual(cm, class_seq[i][-1])

        class_stream = ClassStream(train_dataset, test_dataset,
                                   class_batch_seq=[(0, [0, 3], {0: 0, 3: 0}), (1, [1, 2], {1: 1, 2: 1})])
        class_seq = [
            (0, [0, 3], {0: 0, 3: 0}),
            (1, [1, 2], {1: 1, 2: 1})
        ]
        batches_inputs = [
            [[2.0, 2.0], [1.0, 1.0]],
            [[3.0, 3.0], [0.0, 0.0], [4.0, 4.0], [5.0, 5.0]]
        ]

        self.__test_class_stream(class_stream, class_seq, batches_inputs)
        for i, (_, _, cm) in enumerate(class_stream.get_train_data()): self.assertDictEqual(cm, class_seq[i][-1])

    def test_class_stream_init(self):
        inputs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
        labels = [2, 2, 0, 1, 2, 2, 1]
        train_dataset = test_dataset = DummyDataset(inputs, labels)

        class_stream = ClassStream(train_dataset, test_dataset, class_size=1, init_data={1: 0.5, 2: 0.5})
        class_seq = [
            (0, [0]),
            (1, [1]),
            (2, [2])
        ]
        batches_inputs = [
            [[2.0, 2.0]],
            [[6.0, 6.0]],
            [[4.0, 4.0], [5.0, 5.0]]]
        self.__test_class_stream(class_stream, class_seq, batches_inputs)

        init_class_concept_mapping, init_data = class_stream.get_init_data()
        init_data_loader = DataLoader(init_data, batch_size=len(init_data))
        inputs, labels = next(iter(init_data_loader))

        np.testing.assert_equal(inputs, np.array([[3.0, 3.0], [0.0, 0.0], [1.0, 1.0]]))
        np.testing.assert_equal(labels, np.array([1, 2, 2]))
        self.assertDictEqual(init_class_concept_mapping, {1: 1, 2: 2})

    @staticmethod
    def __test_class_stream(class_stream, class_seq, batches_inputs):
        i = 0
        for class_idx, class_batch_data, _ in class_stream.get_train_data():
            class_batch_data_loader = DataLoader(class_batch_data, batch_size=2)
            batch_classes, batch_inputs = set(), []

            for inputs, labels in class_batch_data_loader:
                batch_classes.update(labels.tolist())
                batch_inputs.extend(inputs.tolist())

            np.testing.assert_equal(class_idx, class_seq[i][0])
            np.testing.assert_equal(batch_classes, set(class_seq[i][1]))
            np.testing.assert_equal(batch_inputs, batches_inputs[i])
            i += 1


class DummyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
