from typing import Dict
from torch.utils.data import Dataset
from torch.tensor import Tensor
from tqdm import tqdm
import re
import numpy as np


class ArffDataset(Dataset):
    def __init__(self, path):
        stream = read_arff_stream(path)
        data = stream.stream
        self.rows, self.labels = data[:, :-1], data[:, -1]
        self.n = len(self.labels)

    def __getitem__(self, index):
        return self.rows[index], self.labels[index]  # todo: Tensor()?

    def __len__(self):
        return self.n


class StreamDefinition:
    def __init__(self, name: str, att_num: int, cls_num: int, cls_map: Dict, length: int, gran: int = 1):
        self.name = name
        self.att_num = att_num
        self.cls_num = cls_num
        self.cls_map = cls_map
        self.length = length
        self.gran = gran


class StreamMeta:
    def __init__(self, labeled: bool, predicted_correct: bool):
        self.labeled = labeled
        self.predicted_correct = predicted_correct


class Stream:
    def __init__(self, stream: np.ndarray, stream_def: StreamDefinition, stream_meta: StreamMeta = None):
        self.stream = stream
        self.stream_def = stream_def
        self.stream_meta = stream_meta


def read_arff_stream(path: str):
    tqdm.write('Reading: {0}'.format(path))
    file = open('{0}'.format(path), 'r')
    stream_def = read_arff_header(file)
    [stream_data, stream_meta_data] = read_arff_data(file, stream_def)

    return Stream(stream_data, stream_def, stream_meta_data)


def read_arff_header(file):
    file_content = list(filter(lambda row: row, file.read().split('\n')))
    name = file_content[0].lower().replace('@relation', '').split('/')[-1].replace('\'', '').strip().upper()

    i = 1
    while '@attribute' in file_content[i].lower():
        i += 1

    att_num = i - 2
    classes = re.search('{(.*)}', file_content[i - 1]).group(1).split(',')
    cls_map = {classes[j].strip(): j for j in range(0, len(classes))}
    cls_num = len(classes)
    length = len(file_content) - att_num - 3

    stream_def = StreamDefinition(name, att_num, cls_num, cls_map, length)
    return stream_def


def read_arff_data(file, stream_def: StreamDefinition):
    file.seek(0)
    file_data_content = list(filter(lambda x: x and '@' not in x, file.read().split('\n')))

    data = list(map(lambda row: row_str_to_values(row, stream_def), file_data_content))
    [stream_data, stream_meta_data] = zip(*data)

    return [np.array(stream_data), np.array(stream_meta_data)]


def row_str_to_values(row: str, stream_def: StreamDefinition):
    elements = row.split(',')
    atts = list(map(lambda x: float(x), elements[0:-1]))
    meta_data = []
    cls_col = elements[-1]
    cls = stream_def.cls_map[cls_col]

    return [atts + [cls], meta_data]



