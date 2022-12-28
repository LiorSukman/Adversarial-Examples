import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from train_mnist import Net

EPS = [0, .05, .1, .15, .2, .25, .3]
BATCH_SIZE = 1


# The code in this file was adapted from the notebook that can be found here:
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test_eps(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        
        # Zero all existing gradients
        model.zero_grad()

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Calculate gradients of model in backward pass
        loss.backward()
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def plot_graphs(accuracies, examples):
    plt.figure(figsize=(5, 5))
    plt.plot(EPS, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(EPS)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(EPS), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(EPS[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


def main(model_path):
    print('downloading dataset...')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./dataset', train=False, download=True, transform=transform)

    test_kwargs = {'batch_size': BATCH_SIZE, 'shuffle':True}
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    print('creating model...')
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in EPS:
        acc, ex = test_eps(model, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plot_graphs(accuracies, examples)

if __name__ == '__main__':
    main('trained_models/mnist_cnn_best_1672124664.pt')
