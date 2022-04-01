import torchvision
import torch
from torchvision import transforms
from torch.utils.data import *
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
MAX_WORDS = 10000  # imdb’s vocab_size 即词汇表大小
MAX_LEN = 200      # max length
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

class IMDB_train(torch.utils.data.Dataset):
    def __init__(self, remain=None):
        self.train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
        if remain is not None:
            self.remain = remain
        else:
            self.remain = list(range(len(x_train)))
    def __getitem__(self, index):
        data,label = self.train_data[index]
        return index, data, label
    def __len__(self):
        return len(self.remain)

class IMDB_test(torch.utils.data.Dataset):
    def __init__(self):
        self.test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    def __getitem__(self, index):
        data,label = self.test_data[self.remain[index]]
        return self.remain[index], data, label
    def __len__(self):
        return len(x_test)

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
    def __init__(self, dataset_path='./data/', transformations=None, should_download=True, remain=None):
        self.dataset_train = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                          download=should_download, transform=transform_cifar10)
        self.transformations = transformations
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
