import torchvision
import torch
from torchvision import transforms


class MNIST_train(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data/', transformations=None, should_download=True, remain=None):
        self.dataset_train = torchvision.datasets.MNIST(root=dataset_path,
                                                        transform=transform,
                                                        train=True,
                                                        download=should_download)
        if remain is None:
            self.remain = list(range(len(self.dataset_train)))
        else:
            self.remain = remain
        self.transformations = transformations

    def __getitem__(self, index):
        (img, label) = self.dataset_train[self.remain[index]]
        if self.transformations is not None:
            return self.transformations(img), label
        return self.remain[index], img, label

    def __len__(self):
        return len(self.remain)

class MNIST_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data/', transformations=None, should_download=True):
        self.dataset_train = torchvision.datasets.MNIST(root=dataset_path,
                                                        transform=transform,
                                                        train=False,
                                                        download=should_download)
        self.transformations = transformations

    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label
        return index, img, label

    def __len__(self):
        return len(self.dataset_train)

class Cifar10_train(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data/', transformations=None, should_download=True):
        self.dataset_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                        download=should_download, transform=transform_cifar10)
        self.transformations = transformations

    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label
        return index, img, label

    def __len__(self):
        return len(self.dataset_train)

class Cifar10_test(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data/', transformations=None, should_download=True):
        self.dataset_train = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                        download=should_download, transform=transform_cifar10)
        self.transformations = transformations

    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label
        return index, img, label

    def __len__(self):
        return len(self.dataset_train)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

transform_cifar10 = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

