import tensorflow as tf
import numpy as np
import sklearn
from torch.utils.data import DataLoader, Dataset

from utils.plt_utils import PlotUtils as pu


class TBScalars:

    @staticmethod
    def write_epoch_result(tasks_acc, epoch, stream_label, i):
        tf.summary.scalar(f'{stream_label}#EPOCHS/C{i}', sum(tasks_acc) / len(tasks_acc), epoch,
                          description='x=epochs, y=overall accuracy')
        tf.summary.flush()

    @staticmethod
    def write_tasks_results(stream_label, tasks_acc, i):
        for j, acc in enumerate(tasks_acc):
            tf.summary.scalar(f'{stream_label}/C{j}', acc, i,
                              description='x=class ids, y=accuracy for a given Ci')  # todo: add configurable class-aggregation?

        tf.summary.scalar(f'ALL/{stream_label}', sum(tasks_acc) / len(tasks_acc), i,
                          description='x=class ids, y=overall accuracy')
        tf.summary.flush()


class TBImages:

    @staticmethod
    def write_test_data(data: Dataset, i: int, stream_label: str, cls_names: list):
        loader = DataLoader(data, batch_size=100, shuffle=True)

        images, labels = next(iter(loader))
        images = np.transpose(images.reshape(*images.shape), (0, 2, 3, 1))
        figure = pu.create_image_grid(images, labels, cls_names)

        tf.summary.image(f'{stream_label}#EXAMPLES', pu.fig_to_image(figure), step=i)
        tf.summary.flush()

    @staticmethod
    def write_confusion_matrices(labels, preds, i, stream_label):
        cm = sklearn.metrics.confusion_matrix(labels, preds)
        cm[np.isnan(cm)] = 0.0

        figure = pu.create_confusion_matrix(cm, class_names=[f'C{k}' for k in range(len(cm))])

        tf.summary.image(f'{stream_label}#CONF-MATS', pu.fig_to_image(figure), step=i)
        tf.summary.flush()
