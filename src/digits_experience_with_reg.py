import pickle
import time
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

from src.regularization.regularizer import L1Regularizer, L2Regularizer, ElasticNetRegularizer, \
    GroupSparseLassoRegularizer, GroupLassoRegularizer


def model_test(model, dataset, batch_size, regularizer_loss, param_name='', all_param_regularized=True, use_gpu=False):
    dataset.transform = ToTensor()
    loader, _ = train_valid_loaders(dataset, batch_size)
    return validate_with_reg(model, loader, regularizer_loss, param_name, all_param_regularized, use_gpu)


def validate_with_reg(model, val_loader, regularizer_loss, param_name='', all_param_regularized=True, use_gpu=False):
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
            if all_param_regularized:
                loss_val = regularizer_loss.regularized_all_param(reg_loss_function=loss_val)
            else:
                assert param_name != '', 'you must specified the name of the parameters to be regularized'
                for model_param_name, model_param_value in model.named_parameters():
                    if model_param_name == param_name:
                        loss_val = regularizer_loss.regularized_param(param_weights=model_param_value,
                                                                      reg_loss_function=loss_val)
            val_loss.append(loss_val.item())
            # val_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train_with_regularizer(model, dataset, n_epoch, batch_size, learning_rate, regularizer_loss, param_name='',
                           all_param_regularized=True, use_gpu=False):
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

            if all_param_regularized:
                loss = regularizer_loss.regularized_all_param(reg_loss_function=loss)
            else:
                assert param_name != '', 'you must specified the name of the parameters to be regularized'
                for model_param_name, model_param_value in model.named_parameters():
                    if model_param_name == param_name:
                        loss = regularizer_loss.regularized_param(param_weights=model_param_value,
                                                                  reg_loss_function=loss)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate_with_reg(model, train_loader, regularizer_loss, param_name,
                                                  all_param_regularized, use_gpu)
        val_acc, val_loss = validate_with_reg(model, val_loader, regularizer_loss, param_name,
                                              all_param_regularized, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history


def main_digits_with_regularizer(type_of_reg='l1', result_file='l1_NN', param_name='', all_param_regularized=True,
                                 use_gpu=False):
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

    lr_list_param = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5]
    n_epoch_list = [200]
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

    model = FullyConnectedNN(input_dim=64, output_dim=10, activation_function='Softmax', layers_sizes=[40, 20])
    saving_dict = defaultdict(dict)
    for i, param in enumerate(params):
        if type_of_reg == 'l1':
            reg_loss = L1Regularizer(model=model, lambda_reg=0.01)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'l2':
            reg_loss = L2Regularizer(model=model, lambda_reg=0.01)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'EL':
            reg_loss = ElasticNetRegularizer(model=model, lambda_reg=0.01)
            reg_loss.lambda_reg = param['ld_reg']
            reg_loss.alpha_reg = param['alpha_reg']
        if type_of_reg == 'GL':
            reg_loss = GroupLassoRegularizer(model=model, lambda_reg=0.01)
            reg_loss.lambda_reg = param['ld_reg']
        if type_of_reg == 'SGL':
            reg_loss = GroupSparseLassoRegularizer(model=model, lambda_reg=0.01)
            reg_loss.lambda_reg = param['ld_reg']
        history_trained = train_with_regularizer(model=model,
                                                 n_epoch=param['n_epoch'],
                                                 dataset=digits_train,
                                                 batch_size=batch_size,
                                                 learning_rate=param['lr'],
                                                 regularizer_loss=reg_loss,
                                                 param_name=param_name,
                                                 all_param_regularized=all_param_regularized,
                                                 use_gpu=use_gpu)
        # history_trained.display()
        tested_model = model_test(model=model, dataset=digits_test, batch_size=batch_size, regularizer_loss=reg_loss,
                                  param_name=param_name, all_param_regularized=all_param_regularized, use_gpu=use_gpu)
        print(tested_model)
        saving_dict['param_{}'.format(i)] = [param, history_trained.history, tested_model]

    result_file = result_file + time.strftime("%Y%m%d-%H%M%S") + ".pck"
    with open(result_file, 'wb') as f:
        pickle.dump(saving_dict, f)


if __name__ == '__main__':
    # On va tester avec une seule combination et juste la L2 pour verifier notre regularization: L2 and weight decay
    # should give the same thing
    main_digits_with_regularizer(type_of_reg='l2', result_file='l2_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)
    main_digits_with_regularizer(type_of_reg='l1', result_file='l1_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)
    main_digits_with_regularizer(type_of_reg='l2', result_file='l2_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)
    main_digits_with_regularizer(type_of_reg='EL', result_file='EL_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)
    main_digits_with_regularizer(type_of_reg='GL', result_file='GL_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)
    main_digits_with_regularizer(type_of_reg='SGL', result_file='SGL_NN', param_name='',
                                 all_param_regularized=True, use_gpu=True)



