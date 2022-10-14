from torchvision import transforms
from PIL import Image
import torch
import random
import numpy as np

#imagenet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def Transform(resize, imagesize) :
    """
    Default transform from Image to normalized tensor
    Args:
        resize (int): Resize shape
        imagesize (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(imagesize),
                        transforms.ToTensor(),                        
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform

def GT_Transform(resize, imagesize) :
    """
    Default transform from ground truth image to tensor (not normalize)
    Args:
        resize (int): Resize shape
        imagesize (int): CenterCrop shape
    """
    transform = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(imagesize),
                        transforms.ToTensor()])
    return transform

def INV_Normalize() :
    """
    Inverse normalize from normalized tensor
    """
    transform = transforms.Normalize(mean = - torch.tensor(IMAGENET_MEAN) / torch.tensor(IMAGENET_STD), std = 1 / torch.tensor(IMAGENET_STD))
    return transform

def generate_cutpaste_info(imagesize, area_ratio, aspect_ratio):
    img_area = imagesize * imagesize
    patch_area = random.uniform(*area_ratio) * img_area
    patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
    patch_w  = int(np.sqrt(patch_area*patch_aspect))
    patch_h = int(np.sqrt(patch_area/patch_aspect))
    patch_left, patch_top = random.randint(0, imagesize - patch_w), random.randint(0, imagesize - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    # new location
    paste_left, paste_top = random.randint(0, imagesize - patch_w), random.randint(0, imagesize - patch_h)
    cutpaste_info = (patch_left, patch_top, patch_right, patch_bottom, paste_left, paste_top)
    
    return cutpaste_info

def cutpaste(imagesize, area_ratio, aspect_ratio, maxcut, transform=None, rotation=False) :
    num_cut = random.randint(1, maxcut)
    
    def _cutpaste(image, gt):
        aug_image = image.clone()
        aug_gt = gt.clone()
        
        for idx in range(num_cut) :
            patch_left, patch_top, patch_right, patch_bottom, paste_left, paste_top = generate_cutpaste_info(imagesize, area_ratio, aspect_ratio)        
        
            patch = image[:, patch_left:patch_right, patch_top:patch_bottom]
            
            if transform:
                patch= transform(patch)

            if rotation:
                random_rotate = random.uniform(*rotation)
                patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
                mask = patch.split()[-1]

            # new location
            aug_image[:, paste_left : paste_left + patch.shape[1], paste_top : paste_top + patch.shape[2]] = torch.mean(patch)
            aug_gt[:, paste_left : paste_left + patch.shape[1], paste_top : paste_top + patch.shape[2]] = 1
        
        return aug_image, aug_gt

    return _cutpaste