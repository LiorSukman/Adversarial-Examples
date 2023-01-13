import numpy as np
import torch
import torch.nn.functional as F

def fgsm_attack(images, epsilons, data_grad, p=1.0):
    images, data_grad = images.unsqueeze(dim=0), data_grad.unsqueeze(dim=0)
    # some validation of the input
    if isinstance(epsilons, np.ndarray):
        assert len(epsilons.shape) == 5 # dimension of image + 1
        epsilons.astype(np.float32)
    else:
        assert isinstance(epsilons, float)
        epsilons = np.array([[[[[epsilons]]]]], dtype=np.float32)

    # Collect the element-wise sign of the data gradient
    mask = torch.reshape(torch.Tensor(np.random.binomial(1, p, len(images))), (1, -1, 1, 1, 1))
    sign_data_grad = (data_grad.sign() * mask).repeat(epsilons.shape)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images.repeat(epsilons.shape) + sign_data_grad * epsilons
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    # Return the perturbed image
    return perturbed_images.squeeze(dim=0)

def replacement_pipeline(images, targets, epsilon, p, model, device):
    """
    Pipeline to replace some of the inputs by the perturbed images based on the FGSM.
    This is a bit different than the approach presented in Goodfellow et al., 2014. Instead of always using both 
    representations (original and perturbed) to update the weights for every input image, we are using only the 
    perturbed images in p of the cases and the original in the others. This was proved to be efficient (see 
    notebook - Goodfellow2014_exp.ipynb experiment 7) while not needing to calculate the loss an extra time.
    """
    training = model.training
    model.eval()
    images.requires_grad = True
    images.grad = None
    output = model(images)      
    model.zero_grad()
    loss = F.nll_loss(output, targets)
    loss.backward()
    data_grad = images.grad.data
    images.requires_grad = False
    if training:
        model.train()
    pert_imgaes = fgsm_attack(images.cpu(), epsilon, data_grad.cpu(), p).to(device)
    return pert_imgaes

def fgsm_regularization(images, targets, epsilon, alpha, model, device):
    pert_images = replacement_pipeline(images, targets, epsilon, 1, model, device)
    output_org = model(images)
    output_pert = model(pert_images)
    loss = alpha * (F.nll_loss(output_org, targets)) + (1 - alpha) * (F.nll_loss(output_pert, targets))
    return loss

def ll_fgsm(images, epsilon, p, model, device, targets=None):
    """
    Similar to replacement_pipeline just here the noise is generated not based on the correct label, but rather based
    on the least likely prediction according to the model (assuming it is already somewhat trained). This fits the idea
    presented in Kurakin et al., 2017 without the iterative part.
    """
    training = model.training
    model.eval()
    images.requires_grad = True
    images.grad = None
    output = model(images)
    ll_targets = targets if targets is not None else output.argmin(dim=1)
    model.zero_grad()
    loss = F.nll_loss(output, ll_targets)
    loss.backward()
    data_grad = -1 * images.grad.data
    images.requires_grad = False
    if training:
        model.train()
    pert_imgaes = fgsm_attack(images.cpu(), epsilon, data_grad.cpu(), p).to(device)
    return pert_imgaes

def iter_adv(images, targets, epsilon, model, ll=False, p=1.0, device='cpu', dynamic_range=255, mask=None):
    assert isinstance(epsilon, float)  # epsilon here represents maximal change, no need to pass more than 1 value
    epsilon_dr = int(epsilon * dynamic_range)
    if mask is None:
        mask = np.random.binomial(1, p, len(images)) == 1.0
    p_placeholder = 1.0
    eps_placeholder = 1.0 / dynamic_range
    pert_images = images
    for i in range(epsilon_dr):
        if not ll:
            pert_images = replacement_pipeline(pert_images, targets, eps_placeholder, p_placeholder, model, device)
        else:
            pert_images = ll_fgsm(pert_images, eps_placeholder, p_placeholder, model, device, targets=targets)
    pert_images[~mask] = images[~mask]
    return pert_images, mask
    

class FGSMTransform:
    """
    Apply FGSM transformation on the data before training based on another model
    """

    def __init__(self, p, epsilon, path, model):
        self.p = p
        self.epsilon = epsilon

        device = torch.device('cpu')
        model = model.to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        self.model = model

    def __call__(self, img, target):
        assert isinstance(img, torch.Tensor)
        if torch.rand(1) < self.p:
            img = torch.unsqueeze(img, 0)
            target = torch.tensor([target])
            img.requires_grad = True
            output = self.model(img)        
            self.model.zero_grad()
            loss = F.nll_loss(output, target)
            loss.backward()
            data_grad = img.grad.data
            img.requires_grad = False
            pert_img = torch.squeeze(fgsm_attack(img, self.epsilon, data_grad), dim=0)
            return pert_img

        return img            