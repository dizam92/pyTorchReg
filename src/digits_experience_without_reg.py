import pickle
import time
import os
import numpy as np
from glob import glob
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from sklearn.metrics import accuracy_score
from src.deeplib.datasets import load_cifar10, load_mnist, load_digits_sklearn
from src.deeplib.history import History
from src.deeplib.data import train_valid_loaders
from src.deeplib.net import MnistNet, CifarNet, FullyConnectedNN
from src.utils import load_file


def model_test(model, dataset, batch_size, use_gpu=False):
    dataset.transform = ToTensor()
    loader, _ = train_valid_loaders(dataset, batch_size)
    return validate_without_reg(model, loader, use_gpu)


def validate_without_reg(model, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(val_loader):

            inputs, targets = batch['data'], batch['target']
            inputs = inputs.type(torch.FloatTensor)
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)

            predictions = output.max(dim=1)[1]

            loss_val = criterion(output, targets)

            val_loss.append(loss_val.item())
            # val_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train_without_regularizer(model, dataset, n_epoch, batch_size, learning_rate, weight_decay=None,
                              use_weight_decay=False, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    if use_weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    dataset.transform = ToTensor()
    train_loader, val_loader = train_valid_loaders(dataset, batch_size=batch_size)

    for i in range(n_epoch):
        model.train()
        for j, batch in enumerate(train_loader):
            inputs, targets = batch['data'], batch['target']
            inputs = inputs.type(torch.FloatTensor)
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate_without_reg(model, train_loader, use_gpu)
        val_acc, val_loss = validate_without_reg(model, val_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history


def main_digits_without_regularizer(weight_decay=False, result_file='fc_normal', use_gpu=False):
    """
    Run the simplest algorithm to compare the action of the regularization
     """
    digits_train, digits_test = load_digits_sklearn()
    batch_size = 300
    weight_decay_list = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    lr_list_param = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    n_epoch_list = [200, 300]

    # lr_list_param = [0.01]
    # n_epoch_list = [200]
    # weight_decay_list = [0.01]
    if weight_decay:
        params = list(ParameterGrid({'lr': lr_list_param,
                                     'n_epoch': n_epoch_list,
                                     'weight_decay': weight_decay_list
                                     }))
    else:
        params = list(ParameterGrid({'lr': lr_list_param,
                                     'n_epoch': n_epoch_list
                                     }))

    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='ReLU', layers_sizes=[40, 20])
    saving_dict = defaultdict(dict)
    for i, param in enumerate(params):
        if weight_decay:
            history_trained = train_without_regularizer(model=model,
                                                        n_epoch=param['n_epoch'],
                                                        dataset=digits_train,
                                                        batch_size=batch_size,
                                                        learning_rate=param['lr'],
                                                        weight_decay=param['weight_decay'],
                                                        use_gpu=use_gpu)
        else:
            history_trained = train_without_regularizer(model=model,
                                                        n_epoch=param['n_epoch'],
                                                        dataset=digits_train,
                                                        batch_size=batch_size,
                                                        learning_rate=param['lr'],
                                                        use_gpu=use_gpu)

        # history_trained.display()
        tested_model = model_test(model=model, dataset=digits_test, batch_size=batch_size, use_gpu=use_gpu)
        print(tested_model)
        saving_dict['param_{}'.format(i)] = [param, history_trained.history, tested_model]

    result_file = result_file + time.strftime("%Y%m%d-%H%M%S") + ".pck"
    with open(result_file, 'wb') as f:
        pickle.dump(saving_dict, f)


if __name__ == '__main__':
    main_digits_without_regularizer(weight_decay=False, result_file='fc_normal_', use_gpu=True)
    main_digits_without_regularizer(weight_decay=True, result_file='fc_normal_with_weight_decay', use_gpu=True)



