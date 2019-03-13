import torch
import torch.nn as nn
import torch.optim as optim
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


def train(model, dataset, n_epoch, batch_size, learning_rate, regularizer_loss, all_param_regularized=True,
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


if __name__ == '__main__':
    mnist_train, mnist_test = load_mnist()
    # digits_train, digits_test = load_digits_sklearn()
    batch_size = 128
    lr = 0.01
    n_epoch = 10
    ld_reg = 0.01
    model = FullyConnectedNN(input_dim=784, output_dim=10, activation_function='ReLU', layers_sizes=[40, 20])
    l1_reg_loss = L1Regularizer(name=None, model=model, lambda_reg=ld_reg)
    history_trained = train(model=model, n_epoch=n_epoch, dataset=mnist_train, batch_size=batch_size, learning_rate=lr,
                            regularizer_loss=l1_reg_loss, all_param_regularized=True, use_gpu=False)
    history_trained.display()

    tested_model = test(model=model, dataset=mnist_test, batch_size=batch_size, regularizer_loss=l1_reg_loss,
                        all_param_regularized=True, use_gpu=False)
    print(tested_model)
