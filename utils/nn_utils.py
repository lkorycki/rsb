import functools
import operator
import os
import torch
from torch.utils import model_zoo


pytorch_models_root = './pytorch_models'


class NeuralNetUtils:

    @staticmethod
    def reset_model(model: torch.nn.Module):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @staticmethod
    def flat_num(feature_extractor, in_size):
        return functools.reduce(operator.mul, list(feature_extractor(torch.rand(1, *in_size)).shape))

    @staticmethod
    def load_model(root: str, name: str):
        model_path = os.path.join(root, f'{name.lower()}.pth')

        if not os.path.exists(model_path):
            model = model_zoo.load_url(model_urls[name])
            torch.save(model.state_dict(), model_path)

        return torch.load(model_path)


model_urls = {
    'MNIST': '',
}


model_creators = {
    'MNIST': lambda: NeuralNetUtils.load_model(pytorch_models_root, 'MNIST')
}


def get(model_name: str):
    return model_creators[model_name]()
