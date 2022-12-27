import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from os.path import isdir
from os import mkdir
import time
import numpy as np

PATIENCE = 20
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1_000
N_EPOCH = 200
LR = 1.0
GAMMA = 0.7
SEED = 42
LOG_INT = 100
SAVE_MODEL = 'trained_models/'
LOAD_MODEL = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2704, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, verbose=True, name='Test set'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print(f'\n{name}: Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f})')
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    torch.manual_seed(SEED)

    device = "mps" if getattr(torch, "has_mps", False) else "cpu" # device to run the model on

    # organize parsed data
    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}
    if device != "cpu":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # get datasets and create loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    dataset1 = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
    train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000], generator=torch.Generator().manual_seed(42))
    dataset2 = datasets.MNIST('./dataset', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # create model, initialize optimizer
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

    dev_losses = []
    train_losses = []
    dev_accuracy = []
    train_accuracy = []
    
    if not isdir(SAVE_MODEL):
        mkdir(SAVE_MODEL)

    if LOAD_MODEL is None:  # don't need to load
        best_epoch = 0
        best_loss = float('inf')
        start_time = time.time()
        # run training
        for epoch in range(1, N_EPOCH + 1):
            train(model, device, train_loader, optimizer, epoch, LOG_INT)
            train_loss, train_acc = test(model, device, train_loader, verbose=False)
            dev_loss, dev_acc = test(model, device, dev_loader, name='Dev set')
            dev_losses.append(dev_loss)
            dev_accuracy.append(dev_acc)
            train_losses.append(train_loss)
            train_accuracy.append(train_acc)
            if dev_loss < best_loss:  # found better epoch
                best_loss = dev_loss
                best_epoch = epoch
                torch.save(model.state_dict(), SAVE_MODEL + 'mnist_cnn_temp.pt')
            if best_epoch + PATIENCE <= epoch:  # no improvment in the last PATIENCE epochs
                print(f'No improvement was done in the last {PATIENCE} epochs, breaking...')
                break
        end_time = time.time()
        print('Training took %.3f seconds' % (end_time - start_time))
        print(f'Best model was achieved on epoch {best_epoch}')
        model.load_state_dict(torch.load(SAVE_MODEL + 'mnist_cnn_temp.pt'))  # load model from best epoch

        epochs = np.arange(1, len(dev_losses) + 1)
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, dev_losses, 'b-', label='dev loss')
        ax1.plot(epochs, train_losses, 'b--', label='train loss')
        ax1.tick_params(axis='y', labelcolor = color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(epochs, dev_accuracy, 'r-', label='dev accuracy')
        ax2.plot(epochs, train_accuracy, 'r--', label='train accuracy')
        ax2.tick_params(axis='y', labelcolor = color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend()
        ax2.legend()
        plt.title('Classifying CNN training - Loss and Accuracy vs. Epoch')
        plt.show()
    else:  # need to load
        model.load_state_dict(torch.load(LOAD_MODEL))

    print('Testing test set...')
    test(model, device, test_loader)


if __name__ == '__main__':
    main()