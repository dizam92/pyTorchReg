import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from src.deeplib.history import History
from src.deeplib.data import train_valid_loaders


def validate(model, val_loader, use_gpu=False):
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

            val_loss.append(criterion(output, targets).item())
            true.extend(targets.data.cpu().numpy().tolist())
            pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train(model, dataset, n_epoch, batch_size, learning_rate, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, use_gpu)
        val_acc, val_loss = validate(model, val_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    return history
