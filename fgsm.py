import numpy as np
import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilons, data_grad):
    # some validation of the input
    if type(epsilons) == np.ndarray:
        assert len(epsilons.shape) == 4 # dimension of image
        epsilons.astype(np.float32)
    else:
        assert isinstance(epsilons, float)
        epsilons = np.array([[[[epsilons]]]], dtype=np.float32)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign().repeat(epsilons.shape)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = image.repeat(epsilons.shape) + sign_data_grad * epsilons
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    # Return the perturbed image
    return perturbed_images

class FGSMTransform:
    """Apply FGSM on the data"""

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