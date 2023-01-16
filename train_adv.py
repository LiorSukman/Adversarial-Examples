import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
from train_mnist import Net

PATIENCE = 20
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1_000
N_EPOCH = 50
LR = 1.0
SEED = 42
LOG_INT = 100
SAVE_MODEL = 'trained_models/'
LOAD_MODEL = None
ORG_MODEL_PATH = 'trained_models/mnist_cnn_best_1672124664.pt'

PIC_DIM = 28
EPS = 0.25
COR_EPS = 2e-16

# regularization variables
C = 1
GAMMA = 0.0
THR = 0.0


def custom_loss(output, noise, target, cnn):
    preds = torch.log(1 - torch.exp(cnn(output))+ COR_EPS)
    loss = C * F.nll_loss(preds, target) + GAMMA * torch.sum(torch.abs(noise)) / len(noise)
    return loss


class ADV_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 784)

    def forward(self, x):
        org = x
        x = self.conv1(x)  # 16,26,26
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16,13,13
        x = torch.flatten(x, 1)  # 2704
        x = self.fc1(x)  # 1024
        x = F.relu(x)
        x = self.fc2(x)  # 784
        x = torch.tanh(torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM)))
        output = torch.clamp(x * EPS + org, 0, 1)
        return output, x

    def generate(self, x, device):
        """
        this function is the same as forward except it uses sign instead of tanh for noise generation
        threshold value THR is also used to limit the noise added
        """
        org = x
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        x = torch.tanh(x).cpu().numpy()
        x = torch.from_numpy(np.where(np.abs(x) > THR, x, 0)).to(device)
        output = torch.clamp(torch.sign(x) * EPS + org, 0, 1)
        return output, x


def train(model, device, train_loader, optimizer, epoch, cnn, log_interval, verbose=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output, noise = model(data)
        loss = custom_loss(output, noise, target, cnn)
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader, cnn, verbose=True, name='Test set'):
    model.eval()
    # values for statistics
    test_loss = 0
    noise_count = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, noise = model(data)

            # update statistics values
            noise_count += torch.sum(torch.abs(noise))
            test_loss += custom_loss(output, noise, target, cnn) * len(data)
            output = cnn(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print(f'{name}: Average loss: {test_loss: .4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%), noise average = {noise_count / len(test_loader.dataset):.4f}\n')

    return test_loss, correct / len(test_loader.dataset)


def main():
    torch.manual_seed(SEED)
    # Currently using cpu instead of mps due to https://discuss.pytorch.org/t/training-doesnt-converge-when-running-on-m1-pro-gpu-mps-device/157918
    device = 'cpu' # "mps" if getattr(torch, "has_mps", False) else "cpu" # device to run the model on

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
    transform = transforms.Compose([transforms.ToTensor()])

    dataset1 = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
    train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000], generator=torch.Generator().manual_seed(42))
    dataset2 = datasets.MNIST('./dataset', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    cnn = Net()
    cnn.load_state_dict(torch.load(ORG_MODEL_PATH))
    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()
    cnn = cnn.to(device)

    # create model, initialize optimizer
    model = ADV_NN().to(device)
    optimizer = optim.Adadelta(model.parameters())

    if not LOAD_MODEL:  # don't need to load
        best_epoch = 0
        best_loss = float('inf')
        start_time = time.time()
        # run training
        for epoch in range(1, N_EPOCH + 1):
            train(model, device, train_loader, optimizer, epoch, cnn, LOG_INT)
            dev_loss, _ = test(model, device, dev_loader, cnn, 'Dev set')
            if dev_loss < best_loss:  # found better epoch
                best_loss = dev_loss
                best_epoch = epoch
                torch.save(model.state_dict(), SAVE_MODEL + f'mnist_cnn_adv_best_{int(start_time)}.pt')

            if best_epoch + PATIENCE <= epoch:  # no improvement in the last PATIENCE epochs
                print('No improvement was done in the last %d epochs, breaking...' % PATIENCE)
                break

        end_time = time.time()
        print('Training took %.3f seconds' % (end_time - start_time))
        print('Best model was achieved on epoch %d' % best_epoch)
        model.load_state_dict(torch.load(SAVE_MODEL + f'mnist_cnn_adv_best_{int(start_time)}.pt'))  # load model from best epoch
    else:  # need to load
        model.load_state_dict(torch.load(LOAD_MODEL))

    print('Testing test set...')
    test(model, device, test_loader, cnn)


if __name__ == '__main__':
    main()