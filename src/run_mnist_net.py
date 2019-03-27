import pickle
import time
import os
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

from src.regularization.regularizer import L1Regularizer, L2Regularizer, ElasticNetRegularizer, \
    GroupSparseLassoRegularizer, GroupLassoRegularizer


def test(model, dataset, batch_size, regularizer_loss, all_param_regularized=True, use_gpu=False):
    dataset.transform = ToTensor()
    loader, _ = train_valid_loaders(dataset, batch_size)
    return validate(model, loader, regularizer_loss, all_param_regularized, use_gpu)


def validate(model, val_loader, regularizer_loss, all_param_regularized=True, use_gpu=False):
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
            # if all_param_regularized:
            #     loss_val = regularizer_loss.loss_all_params_regularized(reg_loss_function=loss_val)
            # else:
            #     loss_val = regularizer_loss.loss_regularized(reg_loss_function=loss_val)
            val_loss.append(loss_val.item())
            # val_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train_with_regularizer(model, dataset, n_epoch, batch_size, learning_rate, regularizer_loss, all_param_regularized=True,
          use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

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
            # if all_param_regularized:
            #     loss = regularizer_loss.loss_all_params_regularized(reg_loss_function=loss)
            # else:
            #     loss = regularizer_loss.loss_regularized(reg_loss_function=loss)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, regularizer_loss, all_param_regularized, use_gpu)
        val_acc, val_loss = validate(model, val_loader, regularizer_loss, all_param_regularized, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history


def train_without_regularizer(model, dataset, n_epoch, batch_size, learning_rate, weight_decay, use_weight_decay=False, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    if use_weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
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

        train_acc, train_loss = validate(model, train_loader, regularizer_loss, all_param_regularized, use_gpu)
        val_acc, val_loss = validate(model, val_loader, regularizer_loss, all_param_regularized, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history


def validate_mnist(model, val_loader, regularizer_loss, all_param_regularized=True, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(val_loader):

            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            output = model(inputs)

            predictions = output.max(dim=1)[1]

            loss_val = criterion(output, targets)
            if all_param_regularized:
                loss_val = regularizer_loss.loss_all_params_regularized(reg_loss_function=loss_val)
            else:
                loss_val = regularizer_loss.loss_regularized(reg_loss_function=loss_val)
            val_loss.append(loss_val.item())
            # val_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train_mnist(model, dataset, n_epoch, batch_size, learning_rate, regularizer_loss, all_param_regularized=True,
          use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

    dataset.transform = ToTensor()
    train_loader, val_loader = train_valid_loaders(dataset, batch_size=batch_size)

    for i in range(n_epoch):
        model.train()
        for j, batch in enumerate(train_loader):
            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            print(type(inputs))
            print(inputs.type())
            print(inputs.shape)
            output = model(inputs)
            loss = criterion(output, targets)
            if all_param_regularized:
                loss = regularizer_loss.loss_all_params_regularized(reg_loss_function=loss)
            else:
                loss = regularizer_loss.loss_regularized(reg_loss_function=loss)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, regularizer_loss, all_param_regularized, use_gpu)
        val_acc, val_loss = validate(model, val_loader, regularizer_loss, all_param_regularized, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history


def main_mnist():
    mnist_train, mmnist_test = load_mnist()
    batch_size = 32
    lr = 0.01
    n_epoch = 10
    ld_reg = 0.01
    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='ReLU', layers_sizes=[40, 20])
    # # DEBUG SESSION
    # for name, param in model.named_parameters():
    #     if name.endswith('weight'):
    #         print(name, param.shape)
    #     elif name.endswith('bias'):
    #         print(name, param)
    # ###########
    # exit()

    l1_reg_loss = L1Regularizer(name=None, model=model, lambda_reg=ld_reg)
    history_trained = train_mnist(model=model, n_epoch=n_epoch, dataset=mnist_train, batch_size=batch_size, learning_rate=lr,
                            regularizer_loss=l1_reg_loss, all_param_regularized=True, use_gpu=False)
    history_trained.display()

    tested_model = test(model=model, dataset=mmnist_test, batch_size=batch_size, regularizer_loss=l1_reg_loss,
                        all_param_regularized=True, use_gpu=False)
    print(tested_model)


def main_digits_with_regularizer(type_of_reg='l1', result_file='l1_NN', all_param_regularized=True, use_gpu=False):
    """ We run the optimization algorithm for 200 epochs, with mini-batches of 300 elements.
     After training, all weights under 10−3 in absolute value are set to 0.
     C'est très obscure ceci et je ne sais pas pourquoi ils le font ...
     """
    digits_train, digits_test = load_digits_sklearn()
    batch_size = 300
    # lr_list_param = [0.01]
    # n_epoch_list = [200]
    # ld_reg_list = [0.01]
    # alpha_reg_list = [0.1]

    lr_list_param = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    n_epoch_list = [200, 300]
    ld_reg_list = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    alpha_reg_list = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    if type_of_reg == 'EL':
        params = list(ParameterGrid({'lr': lr_list_param,
                                     'n_epoch': n_epoch_list,
                                     'ld_reg': ld_reg_list,
                                     'alpha_reg': alpha_reg_list
                                     }))

    else:
        params = list(ParameterGrid({'lr': lr_list_param,
                                     'n_epoch': n_epoch_list,
                                     'ld_reg': ld_reg_list
                                     }))

    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='ReLU', layers_sizes=[40, 20])
    saving_dict = defaultdict(dict)
    for i, param in enumerate(params):
        if type_of_reg == 'l1':
            reg_loss = L1Regularizer(name=None, model=model)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'l2':
            reg_loss = L2Regularizer(name=None, model=model)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'EL':
            reg_loss = ElasticNetRegularizer(name=None, model=model)
            reg_loss.lambda_reg = param['ld_reg']
            reg_loss.alpha_reg = param['alpha_reg']
        if type_of_reg == 'GL':
            reg_loss = GroupLassoRegularizer(name=None, model=model, group_name=None)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'SGL':
            reg_loss = GroupSparseLassoRegularizer(name=None, model=model, group_name=None)
            reg_loss.lambda_reg = param['ld_reg']
        history_trained = train(model=model,
                                n_epoch=param['n_epoch'],
                                dataset=digits_train,
                                batch_size=batch_size,
                                learning_rate=param['lr'],
                                regularizer_loss=reg_loss,
                                all_param_regularized=all_param_regularized,
                                use_gpu=use_gpu)
        # history_trained.display()
        tested_model = test(model=model, dataset=digits_test, batch_size=batch_size, regularizer_loss=reg_loss,
                            all_param_regularized=all_param_regularized, use_gpu=use_gpu)
        print(tested_model)
        saving_dict['param_{}'.format(i)] = [history_trained, tested_model]

    result_file = result_file + time.strftime("%Y%m%d-%H%M%S") + ".pck"
    with open(result_file, 'wb') as f:
        pickle.dump(saving_dict, f)


def main_digits_without_regularizer(weight_decay=False, result_file='fc_normal', all_param_regularized=True, use_gpu=False):
    """
    Run the simplest algorithm to compare the action of the regularization
     """
    digits_train, digits_test = load_digits_sklearn()
    batch_size = 300

    lr_list_param = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    n_epoch_list = [200, 300]
    params = list(ParameterGrid({'lr': lr_list_param,
                                 'n_epoch': n_epoch_list
                                 }))

    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='ReLU', layers_sizes=[40, 20])
    saving_dict = defaultdict(dict)
    for i, param in enumerate(params):
        history_trained = train(model=model,
                                n_epoch=param['n_epoch'],
                                dataset=digits_train,
                                batch_size=batch_size,
                                learning_rate=param['lr'],
                                all_param_regularized=all_param_regularized,
                                use_gpu=use_gpu)
        # history_trained.display()
        tested_model = test(model=model, dataset=digits_test, batch_size=batch_size, regularizer_loss=reg_loss,
                            all_param_regularized=all_param_regularized, use_gpu=use_gpu)
        print(tested_model)
        saving_dict['param_{}'.format(i)] = [history_trained, tested_model]

    result_file = result_file + time.strftime("%Y%m%d-%H%M%S") + ".pck"
    with open(result_file, 'wb') as f:
        pickle.dump(saving_dict, f)


def load_file(directory):
    os.chdir(directory)
    for fichier in glob('pck'):
        print('{}'.format(fichier))
        d = pickle.load(open(fichier, 'rb'))
        print()


if __name__ == '__main__':
    main_digits(type_of_reg='l1', result_file='l1_NN', all_param_regularized=True, use_gpu=False)
    # main_digits(type_of_reg='l2', result_file='l2_NN', all_param_regularized=True, use_gpu=False)
    # main_digits(type_of_reg='EL', result_file='EL_NN', all_param_regularized=True, use_gpu=False)
    # main_digits(type_of_reg='GL', result_file='GL_NN', all_param_regularized=True, use_gpu=False)
    # main_digits(type_of_reg='SGL', result_file='SGL_NN', all_param_regularized=True, use_gpu=False)



