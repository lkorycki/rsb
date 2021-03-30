import torch
import torchvision
from torchsummary import summary

from data.tensor_set import extract_features
from learners.nnet import mnistnet, cifar10_resnet
import data.data_collection as data_col


def extract():
    print('Running')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    extractor = mnistnet('pytorch_models/fashion2.pth', device)
    extractor.eval().to(device)
    dataset = data_col.get('FASHION-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/fashion10-train.pt', device=device)
    dataset = data_col.get('FASHION-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/fashion10-test.pt', device=device)

    extractor = mnistnet(f'pytorch_models/mnist2.pth', device)
    extractor.eval().to(device)
    dataset = data_col.get('MNIST-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/mnist10-train.pt', device=device)
    dataset = data_col.get('MNIST-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/mnist10-test.pt', device=device)

    extractor = cifar10_resnet('pytorch_models/svhn2.pth', device)
    extractor.eval().to(device)
    dataset = data_col.get('SVHN-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-train.pt', device=device)
    dataset = data_col.get('SVHN-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-test.pt', device=device)

    extractor = cifar10_resnet('pytorch_models/cifar10-2.pth', device)
    extractor.eval().to(device)
    dataset = data_col.get('CIFAR10-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar10-train.pt', device=device)
    dataset = data_col.get('CIFAR10-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar10-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet10-2.pth'))
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET10-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet10-train.pt', device=device)
    dataset = data_col.get('IMAGENET10-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet10-test.pt', device=device)

