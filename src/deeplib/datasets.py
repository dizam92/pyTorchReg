from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


def load_mnist(download=False, path='../datasets/mnist'):
    train_dataset = MNIST(path, train=True, download=download)
    test_dataset = MNIST(path, train=False, download=download)
    return train_dataset, test_dataset


def load_cifar10(download=False, path='../datasets/cifar10'):
    train_dataset = CIFAR10(path, train=True, download=download)
    test_dataset = CIFAR10(path, train=False, download=download)
    return train_dataset, test_dataset


def load_digits_sklearn():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    digits = datasets.load_digits()
    digits_train, digits_test,  = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)
    return digits


if __name__ == '__main__':
    mnist = load_mnist(download=True)
    cifar = load_cifar10(download=True)
