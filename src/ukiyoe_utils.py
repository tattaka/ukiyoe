import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn

import albumentations as albu
from albumentations import pytorch as AT
from imgaug import augmenters as iaa

def get_img(x, folder: str='train_images'):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{path}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_tensor(x, **kwargs):
    """
    Convert image.
    """
    return x.transpose(2, 0, 1).astype('float32')

sigmoid = lambda x: 1 / (1 + np.exp(-x))
    
def get_training_augmentation(size=(256, 256)):
    train_transform = [
        albu.Resize(size[0], size[1]), 
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.RandomBrightness(0.1, p=1),
            albu.RandomContrast(0.1, p=1),
            albu.RandomGamma(p=1)
        ], p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
        albu.Cutout(p=0.5)#WIP
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(size=(256, 256)):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(size[0], size[1]),
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)
