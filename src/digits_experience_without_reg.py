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

np.random.seed(42)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
                # inputs = inputs.type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
                targets = targets.cuda()
                model.cuda()

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
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_number_epoch = 0
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
                model.cuda()

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
        if val_acc > best_acc:
            best_number_epoch = i
            best_acc = val_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} at this epoch number {}'.format(best_acc, best_number_epoch))
    model.load_state_dict(best_model_wts)
    return history, model


def main_digits_without_regularizer(weight_decay=False, result_file='fc_normal', use_gpu=False):
    """
    Run the simplest algorithm to compare the action of the regularization
     """
    digits_train, digits_test = load_digits_sklearn()
    batch_size = [32, 64, 128]
    weight_decay_list = [10 ** -1, 10 ** -2, 10 ** -3, 10 ** -3.5, 10 ** -4, 10 ** -5, 10 ** -6, 10 ** -7, 10 ** -8,
                         10 ** -9,
                         10 ** -10]
    lr_list_param = [1e-2, 1e-3, 1e-4, 1e-5]
    n_epoch_list = [200, 300]

    if weight_decay:
        params = list(ParameterGrid({'batch_size': batch_size,
                                     'lr': lr_list_param,
                                     'n_epoch': n_epoch_list,
                                     'weight_decay': weight_decay_list
                                     }))
    else:
        params = list(ParameterGrid({'batch_size': batch_size,
                                     'lr': lr_list_param,
                                     'n_epoch': n_epoch_list
                                     }))

    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='Softmax', layers_sizes=[40, 20])
    model.apply(init_weights)
    best_model = model
    saving_dict = defaultdict(dict)
    for i, param in enumerate(params):
        if weight_decay:
            history_trained, best_model = train_without_regularizer(model=model,
                                                                    n_epoch=param['n_epoch'],
                                                                    dataset=digits_train,
                                                                    batch_size=param['batch_size'],
                                                                    learning_rate=param['lr'],
                                                                    weight_decay=param['weight_decay'],
                                                                    use_gpu=use_gpu)
        else:
            history_trained, best_model = train_without_regularizer(model=model,
                                                                    n_epoch=param['n_epoch'],
                                                                    dataset=digits_train,
                                                                    batch_size=param['batch_size'],
                                                                    learning_rate=param['lr'],
                                                                    use_gpu=use_gpu)

        # history_trained.display()
        tested_model = model_test(model=model, dataset=digits_test, batch_size=param['batch_size'], use_gpu=use_gpu)
        print(tested_model)
        saving_dict['param_{}'.format(i)] = [param, history_trained.history, tested_model]

    torch.save(best_model, '{}_best_model.pt'.format(result_file))
    result_file = result_file + time.strftime("%Y%m%d-%H%M%S") + ".pck"
    with open(result_file, 'wb') as f:
        pickle.dump(saving_dict, f)


if __name__ == '__main__':
    main_digits_without_regularizer(weight_decay=False, result_file='fc_normal_', use_gpu=True)
    main_digits_without_regularizer(weight_decay=True, result_file='fc_normal_with_weight_decay', use_gpu=True)
