# Adapted from
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMnistNet(nn.Module):

    def __init__(self):
        super(DeepMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 150, 3, padding=1)
        self.conv3 = nn.Conv2d(150, 300, 3, padding=1)
        self.conv4 = nn.Conv2d(300, 300, 3, padding=1)
        self.conv5 = nn.Conv2d(300, 150, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(150 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MnistNet(nn.Module):

    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(50 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1)
        self.conv3 = nn.Conv2d(50, 150, 3, padding=1)
        self.fc1 = nn.Linear(150 * 8 * 8, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarNetBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 150, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(150)
        self.fc1 = nn.Linear(2400 , 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation_function='Softmax', layers_sizes=[]):
        """
        :param input_dim: the size of the input dim
        :param output_dim: the size of the prediction (Y.shape)
        :param layers_sizes: list, with the size of each layers of the network
        :param activation_function:
        """
        super(FullyConnectedNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        last_layer_size = input_dim
        top_layers = []
        for layer_size in layers_sizes:
            top_layers.append(nn.Linear(last_layer_size, layer_size))
            top_layers.append(nn.ReLU())
            last_layer_size = layer_size
        assert activation_function in vars(nn.modules.activation), \
            'The activation function {} is not supported.'.format(activation_function)

        self.fc_net = nn.Sequential(
            *top_layers,
            nn.Linear(last_layer_size, output_dim),
            vars(nn.modules.activation)[activation_function]()
        )

    def _forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc_net(x)
        return out

    def forward(self, batch):
        return torch.cat([self._forward(x) for x in batch])

