import collections
import os
import zipfile
from typing import Optional, Callable, Dict
import torch
from typing import Sequence
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import T_co


class IndexDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return data, target, index

    def __len__(self):
        return self.dataset.__len__()


class ClassSubset(Subset):

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], classes: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.cls_map = {c: i for i, c in enumerate(classes)}

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        return img, self.cls_map[target]


class DataUtils:

    @staticmethod
    def create_dataset_subset(dataset: Dataset, classes: list, indices_path: str):
        if os.path.exists(indices_path):
            print(f'Loading indices from {indices_path}')
            return ClassSubset(dataset, torch.load(indices_path), classes)

        indices_per_class = DataUtils.get_class_indices(IndexDataset(dataset))

        indices = []
        for c in classes:
            indices.extend(indices_per_class[c])

        print(f'Writing indices to {indices_path}')
        torch.save(indices, indices_path)

        return ClassSubset(dataset, indices, classes)

    @staticmethod
    def create_dataset(root: str, dir_name: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                       download: bool = False, download_func: Optional[Callable] = None, zip_file: str = None, post_func: Optional[Callable] = None):
        input_folder = os.path.join(root, dir_name)

        if not os.path.exists(input_folder):
            if download and download_func is not None:
                print('Downloading the dataset')
                download_func()

                if os.path.exists(f'{root}/{zip_file}'):
                    print(f'Extracting from {zip_file}')

                    with zipfile.ZipFile(f'{root}/{zip_file}', 'r') as zip_ref:
                        zip_ref.extractall(root)
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        if post_func is not None:
            post_func(input_folder)

        return ImageFolder(input_folder, transform, target_transform)

    @staticmethod
    def get_class_indices(dataset: IndexDataset) -> Dict[int, list]:
        class_indices = collections.defaultdict(list)

        for inputs, labels, indices in DataLoader(dataset, batch_size=1024, num_workers=4):
            for label, idx in zip(labels.tolist(), indices.tolist()):
                class_indices[label].append(idx)

        return class_indices

