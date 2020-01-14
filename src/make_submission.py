import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
import argparse
from importlib import import_module
import gc
import numpy as np
import pandas as pd

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

import albumentations as albu
from albumentations import pytorch as AT

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import CriterionCallback, OptimizerCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, MixupCallback, AUCCallback, AccuracyCallback

import apex
from apex import amp


from ukiyoe_utils import * 
from dataset import UkiyoeDataset
from optimizers import get_optimizer 
from cls_models import get_model

from sync_batchnorm import convert_model as sbn_convert_model
from pytorch_toolbelt.inference import tta
from my_tta_functions import *

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
torch.backends.cudnn.deterministic = True
np.random.seed(666)
torch.manual_seed(666)


def main(config):
    opts = config()
    path = opts.path
    train_df = pd.read_csv(f'{path}/train.csv')
    
    n_train = len(np.load("../input/ukiyoe-train-imgs.npz")["arr_0"])
    n_test = len(np.load("../input/ukiyoe-test-imgs.npz")["arr_0"])
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')
   
    DEVICE = 'cuda:0'

    model = get_model(model_type = opts.model_type,
                          encoder= opts.encoder,
                          encoder_weights = opts.encoder_weights,
                          metric_branch = opts.metric_branch,
                          middle_activation = opts.mid_activation,
                          last_activation = None,
                          num_classes = opts.num_class
                         )
    model = sbn_convert_model(model)
#     preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    probabilities = np.zeros((n_test, opts.num_class))
    test_ids = np.array(list(range(n_test)))+1
    for i in tqdm.trange(opts.fold_max, desc=f"fold loop", leave=False):
        logdir = f"{opts.logdir}/fold{i}" 
        test_dataset =  UkiyoeDataset(datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        test_loader = DataLoader(test_dataset, batch_size=opts.batchsize, shuffle=False, num_workers=opts.num_workers)
        checkpoint = torch.load(f"{logdir}/checkpoints/best.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        if opts.tta:
            tta_model = tta.TTAWrapper(model, tta.fliplr_image2label)
#             tta_model = tta.TTAWrapper(model, tta.tencrop_image2label, crop_size=(int(opts.img_size[0]*0.9), int(opts.img_size[1]*0.9)))
            tta_model = tta.TTAWrapper(tta_model, bright_hl_image2label)
        else:
            tta_model = model
        model.to(DEVICE)
        count=0
        for i, (batch) in enumerate(tqdm.tqdm(test_loader, desc="batch loop", leave=False)):
            with torch.no_grad():
                probability_batch = tta_model(batch.to(DEVICE))
#             print(probability_batch.cpu().numpy().shape)
            probabilities[count:count+batch.size(0)] += softmax(probability_batch.cpu().numpy())
            count += batch.size(0)
            
    probabilities /= opts.fold_max
    if opts.tta:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_tta_test.npy', probabilities)
    else:
        np.save(f'probabilities/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_test.npy', probabilities)
    labels = probabilities.argmax(axis=1)
    torch.cuda.empty_cache()
    gc.collect()

    del probabilities
    gc.collect()
    
    ############# predict ###################
    df = pd.DataFrame(labels, columns=['y'])
    df.index.name = 'id'
    df.index = df.index + 1
    if opts.tta:
        df.to_csv(f'predicts/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}_tta.csv', float_format='%.5f')
    else:
        df.to_csv(f'predicts/{opts.logdir.split("/")[-1]}_{opts.img_size[0]}x{opts.img_size[1]}.csv', float_format='%.5f')

def softmax(a, axis=1):
    c = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=axis, keepdims=True)
    y = exp_a / sum_exp_a

    return y 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default_cfg')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)