from torchvision import transforms
from PIL import Image
import torch

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def Transform(load_size, input_size) :
    """
    Default transform from Image to normalized tensor
    Args:
        load_size (int): Resize shape
        input_size (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize((load_size, load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train),
                        transforms.FiveCrop(input_size),
                        lambda crops: torch.stack([crop for crop in crops]),
                        ])
    return transform

def GT_Transform(load_size, input_size) :
    """
    Default transform from ground truth image to tensor (not normalize)
    Args:
        load_size (int): Resize shape
        input_size (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize((load_size, load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(input_size)])
    return transform

def INV_Normalize() :
    """
    Inverse normalize from normalized tensor
    """
    transform = transforms.Normalize(mean = - torch.tensor(mean_train) / torch.tensor(std_train), std = 1 / torch.tensor(std_train))
    return transform

