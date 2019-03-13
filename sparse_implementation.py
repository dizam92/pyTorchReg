import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, BestModelRestore
from sklearn.metrics import hamming_loss, accuracy_score, zero_one_loss
from math import ceil


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation_function='ReLu', layers_sizes=[]):
        """
        :param input_dim: the size of the input dim
        :param output_dim: the size of the prediction (Y.shape)
        :param layers_sizes: list, with the size of each layers of the network
        :param activation_function:
        :param lr: learning rate
        """
        super(FullyConnectedNN, self).__init__()
        self.input = input
        self.output_dim = output_dim
        last_layer_size = input_dim
        top_layers = []
        for layer_size in layers_sizes:
            top_layers.append(nn.Linear(last_layer_size, layer_size))
            top_layers.append(nn.ReLU())
            last_layer_size = layer_size
        assert activation_function in vars(torch.nn.modules.activation), \
            'The activation function {} is not supported.'.format(activation_function)

        self.fc_net = nn.Sequential(
            *top_layers,
            nn.Linear(last_layer_size, output_dim),
            vars(torch.nn.modules.activation)[activation_function]()
        )

    def forward(self, x):
        self.fc_net(x)


class GroupSparseNN:
    def __init__(self, input_dim, output_dim, loss_function=None, activation_function='ReLu', layers_sizes=[], lr=0.001):
        super(GroupSparseNN, self).__init__()
        self.lr = lr
        self.fc_network = FullyConnectedNN(input_dim, output_dim, activation_function, layers_sizes)
        if torch.cuda.is_available():
            self.fc_network.cuda()
        if loss_function is None:
            self.loss = nn.MultiLabelSoftMarginLoss()
        else:
            self.loss = loss_function

        optimizer = optim.Adam(self.fc_network.parameters(), lr=self.lr)
        self.model = Model(self.fc_network, optimizer, self.loss)  # Pytoune Encapsulation

    def fit(self, x_train, y_train, x_valid, y_valid, n_epochs=100, batch_size=32,
            log_filename=None, checkpoint_filename=None, with_early_stopping=True):
        """
        :param x_train: training set examples
        :param y_train: training set labels
        :param x_valid: testing set examples
        :param y_valid: testing set labels
        :param n_epochs: int, number of epoch default value 100
        :param batch_size: int, size of the batch  default value 32, must be multiple of 2
        :param log_filename: optional, to output the training informations
        :param checkpoint_filename: optional, to save the model
        :param with_early_stopping: to activate the early stopping or not
        :return: self, the model
        """
        callbacks = []
        if with_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
            callbacks += [early_stopping]
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, factor=1 / 10, min_lr=1e-6)
        best_model_restore = BestModelRestore()
        callbacks += [reduce_lr, best_model_restore]
        if log_filename:
            logger = CSVLogger(log_filename, batch_granularity=False, separator='\t')
            callbacks += [logger]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]

            # self.model.fit(x_train, y_train, x_valid, y_valid,
            #                batch_size=batch_size, epochs=n_epochs,
            #                callbacks=callbacks)
            nb_steps_train, nb_step_valid = int(len(x_train) / batch_size), int(len(x_valid) / batch_size)
            self.model.fit_generator(generator(x_train, y_train, batch_size), steps_per_epoch=nb_steps_train,
                                     valid_generator=generator(x_valid, y_valid, batch_size),
                                     validation_steps=nb_step_valid,
                                     epochs=n_epochs, callbacks=callbacks, )
            return self

    def evaluate(self, x, y, batch_size=16, metrics=None):
        if metrics is None:
            metrics = [hamming_loss]
        valid_gen = generator(x, y, batch=batch_size)
        nsteps = ceil(len(x) / (batch_size * 1.0))
        _, y_pred = self.model.evaluate_generator(valid_gen, steps=nsteps, return_pred=True)
        y_pred = np.concatenate(y_pred, axis=0)
        if torch.cuda.is_available():
            y_true = y.cpu().numpy()
        else:
            y_true = y.numpy()
        res = {metric.__name__: metric(y_true, y_pred.round()) for metric in metrics}
        print('The metrics of the model is: {}'.format(res))
        return res

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)


def generator(x, y, batch):
    n = len(x)
    while True:
        for i in range(0, n, batch):
            yield x[i:i + batch], y[i:i + batch]

# def group_lasso_loss_function():
#
# # if __name__ == '__main__':
