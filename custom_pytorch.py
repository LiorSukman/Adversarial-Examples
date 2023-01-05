import os
from typing import Any, Tuple
from PIL import Image

import torchvision.transforms.functional as F
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets.mnist import MNIST

# Transforms
class CCompose(Compose):
    def __call__(self, img, target=None):  # add target option to Compose call
        for t in self.transforms:
            img = t(img, target)
        return img

class CToTensor(ToTensor):
    def __call__(self, img, target): # Added **args to handle addition of target input 
        return F.to_tensor(img)


class CMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img, target)  # Transforms get target as well

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, MNIST.__name__, "raw")