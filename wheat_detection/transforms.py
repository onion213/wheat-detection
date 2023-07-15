import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def forward(self, image, target):
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


def get_transform(train: bool = False):
    transforms = []
    transforms.append(PILToTensor())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        pass
    return Compose(transforms)
