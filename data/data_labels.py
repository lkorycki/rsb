import csv

from torch.utils.data import Dataset

import data.data_collection as data_col


class DataLabelsUtils:

    @staticmethod
    def get_dataset_labels(dataset: Dataset):
        return dataset.classes if hasattr(dataset, 'classes') else None

    @staticmethod
    def get_imagenet_dataset_labels(dataset):
        m = {}
        with open('data/imagenet_cls_map.txt') as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)

            for d in data:
                m[d[0]] = d[2]

        labels = []
        for k, v in dataset.class_to_idx.items():
            labels.append(m[k])

        return labels


im200_labels = DataLabelsUtils.get_imagenet_dataset_labels(data_col.get('IMAGENET200-TRAIN'))


cls_names = {
    'FASHION': lambda: ['T-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'],
    'CIFAR10': lambda: DataLabelsUtils.get_dataset_labels(data_col.get('CIFAR10-TRAIN')),
    'CIFAR100': lambda: DataLabelsUtils.get_dataset_labels(data_col.get('CIFAR100-TRAIN')),
    'IMAGENET200': lambda: DataLabelsUtils.get_imagenet_dataset_labels(data_col.get('IMAGENET200-TRAIN')),
    'IMAGENET10': lambda: [im200_labels[i] for i in [0, 22, 25, 68, 117, 145, 153, 176, 188, 198]]
}


def get_cls_names(name: str):
    return cls_names[name]()



