import os

import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class TensorDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.rows, self.labels = data[:, :-1], data[:, -1]
        self.n = len(self.labels)

    def __getitem__(self, index):
        return self.rows[index], self.labels[index]

    def __len__(self):
        return self.n


def extract_features(image_dataset: Dataset, extractor: torch.nn.Module, out_path: str, device='cpu'):
    print('Extracting features...')
    loader = torch.utils.data.DataLoader(image_dataset, batch_size=256, num_workers=4)
    n = len(image_dataset)
    init = False
    all_data = None

    i = 0
    for inputs, labels in tqdm(loader):
        with torch.no_grad():
            features = extractor(inputs.to(device)).cpu()
            if not init:
                all_data = torch.zeros((n, features.shape[1] + 1))
                init = True

            all_data[i:i + len(features), :-1] = features
            all_data[i:i + len(features), -1] = labels

        i += len(features)
        print(i)

    print(f'Saving to {out_path}')
    os.makedirs(os.path.join(*out_path.split('/')[:-1]), exist_ok=True)
    torch.save(all_data, out_path)

