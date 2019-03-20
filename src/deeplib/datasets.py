from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np


class DIGITS(Dataset):
    def __init__(self, train=True, return_X_y=True):
        self.train = train
        self.return_X_y = return_X_y
        if self.return_X_y:
            self.digits, self.target = datasets.load_digits(return_X_y=True)
            if self.train:
                self.train_data, _, self.train_labels, _ = train_test_split(self.digits, self.target,
                                                                            test_size=0.33, random_state=42)
                self.train_data = np.asarray([el.reshape(1, -1) for el in self.train_data])

            else:
                _, self.test_data, _, self.test_labels = train_test_split(self.digits, self.target,
                                                                           test_size=0.33, random_state=42)
                self.test_data = np.asarray([el.reshape(1, -1) for el in self.test_data])
        else:
            self.digits = datasets.load_digits()
            if self.train:
                self.train_data, _, self.train_labels, _ = train_test_split(self.digits.data, self.digits.target,
                                                                            test_size=0.33, random_state=42)
                self.train_data = np.asarray([el.reshape(1, -1) for el in self.train_data])
            else:
                _, self.test_data, _, self.test_labels = train_test_split(self.digits.data, self.digits.target,
                                                                           test_size=0.33, random_state=42)
                self.test_data = np.asarray([el.reshape(1, -1) for el in self.test_data])

    def __getitem__(self, item):
        if self.train:
            sample = {'data': self.train_data[item],
                      'target': self.train_labels[item]}
        else:
            sample = {'data': self.test_data[item],
                      'target': self.test_labels[item]}
        return sample

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def load_mnist(download=False, path='../datasets/mnist'):
    train_dataset = MNIST(path, train=True, download=download)
    test_dataset = MNIST(path, train=False, download=download)
    return train_dataset, test_dataset


def load_cifar10(download=False, path='../datasets/cifar10'):
    train_dataset = CIFAR10(path, train=True, download=download)
    test_dataset = CIFAR10(path, train=False, download=download)
    return train_dataset, test_dataset


def load_digits_sklearn():
    train_dataset = DIGITS(train=True, return_X_y=True)
    test_dataset = DIGITS(train=False, return_X_y=True)
    return train_dataset, test_dataset


# class DIGITS_(object):
#     def __init__(self, train=True, return_X_y=True, random_seed=42):
#         self.train = train
#         self.return_X_y = return_X_y
#         if self.return_X_y:
#             self.digits, self.target = datasets.load_digits(return_X_y=True)
#         else:
#             self.digits = datasets.load_digits()
#         self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.digits, self.target, test_size=0.33, random_state=random_seed)
#
#     def __getitem__(self, index):
#         if self.train:
#             digits, target = self.train_data[index], self.train_labels[index]
#         else:
#             digits, target = self.test_data[index], self.test_labels[index]
#         return digits, target
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)


if __name__ == '__main__':
    mnist = load_mnist(download=True)
    cifar = load_cifar10(download=True)
