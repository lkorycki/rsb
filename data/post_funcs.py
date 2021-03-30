import csv
import os
from shutil import copy, rmtree


def imagenet200_val_post(root: str):
    if not os.path.exists(f'{root}/images'):
        return

    with open(f'{root}/val_annotations.txt') as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)

        for d in data:
            file_path, label = d[0], d[1]
            os.makedirs(f'{root}/{label}', exist_ok=True)
            copy(f'{root}/images/{file_path}', f'{root}/{label}')

        rmtree(f'{root}/images')

