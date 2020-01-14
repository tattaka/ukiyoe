import os
import numpy as np

from torch.utils.data import Dataset
import pandas as pd

import albumentations as albu
from albumentations import pytorch as AT

class UkiyoeDataset(Dataset):
    def __init__(self, path:str = '../input', datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
                preprocessing=None):
        self.datatype = datatype
        if datatype != 'test':
            self.imgs = np.load("../input/ukiyoe-train-imgs.npz")["arr_0"]
            self.labels = np.load("../input/ukiyoe-train-labels.npz")["arr_0"]
        else:
            self.imgs = np.load("../input/ukiyoe-test-imgs.npz")["arr_0"]
            self.labels = None
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_id = self.img_ids[idx] - 1
        img = self.imgs[image_id].astype(np.float32) / 255.
        
#         img_mean = np.mean(img, axis=(0, 1), keepdims=True)
#         img_std = np.mean(img, axis=(0, 1), keepdims=True)
#         img = (img - img_mean) / (img_std + 1e-7)
        
        augmented = self.transforms(image=img)
        img = augmented['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']
        if self.datatype != 'test':
            label = self.labels[image_id]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_ids)

class UkiyoePseudoLabelDataset(Dataset):
    def __init__(self, path:str = '../input', pseudo_label_df: pd.DataFrame = None, img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
                preprocessing=None):
        self.imgs = np.load("../input/ukiyoe-test-imgs.npz")["arr_0"]
        self.labels = pseudo_label_df["y"].values
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]
        img = self.imgs[image_id].astype(np.float32) / 255.
        
#         img_mean = np.mean(img, axis=(0, 1), keepdims=True)
#         img_std = np.mean(img, axis=(0, 1), keepdims=True)
#         img = (img - img_mean) / (img_std + 1e-7)
        
        label = self.labels[image_id]
        augmented = self.transforms(image=img)
        img = augmented['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']
        return img, label

    def __len__(self):
        return len(self.img_ids)
    