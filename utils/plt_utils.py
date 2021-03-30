import matplotlib.pyplot as plt
import io
import tensorflow as tf
import itertools
import numpy as np


class PlotUtils:

    @staticmethod
    def create_image_grid(images, labels, cls_names):
        figure = plt.figure(figsize=(20, 20))
        rows, cols = (10, 10) if len(images) == 100 else (5, 5)

        for i in range(min(rows * cols, len(images))):
            cls_idx = labels[i].item()
            plt.subplot(rows, cols, i + 1, title=cls_names[cls_idx] if len(cls_names) > cls_idx else cls_idx)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i])

        return figure

    @staticmethod
    def fig_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)

        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image

    @staticmethod
    def create_confusion_matrix(cm, class_names):
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues if len(cm) > 1 else plt.cm.Blues_r)
        plt.title('Confusion matrix')

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, labels[i, j], horizontalalignment='center', color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure
