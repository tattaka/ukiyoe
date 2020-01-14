import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
import argparse
from importlib import import_module
train_on_gpu = True
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import pytorch as AT

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import CriterionCallback, OptimizerCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, MixupCallback, AUCCallback, AccuracyCallback
from catalyst.dl.callbacks.checkpoint import CheckpointCallback

import apex
from apex import amp


from ukiyoe_utils import * 
from dataset import UkiyoeDataset, UkiyoePseudoLabelDataset
from optimizers import get_optimizer 
from cls_models import get_model

from sync_batchnorm import convert_model as sbn_convert_model
from pytorch_toolbelt.inference import tta
from my_tta_functions import *
# from antialiased_cnns_converter import convert_model as aacnn_convert_model

from my_callbacks import CutMixCallback, LINENotifyCallBack, PixMixCallback


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
torch.backends.cudnn.deterministic = True
np.random.seed(666)
torch.manual_seed(666)
# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def stratified_groups_kfold(df, target, n_splits=5, random_state=0):
    all_groups = pd.Series(df[target])
    if n_splits > 1:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for idx_tr, idx_val in folds.split(all_groups, all_groups):
            idx_tr_new = df.iloc[idx_tr]
            idx_val_new = df.iloc[idx_val]
            print(len(idx_tr_new),  len(idx_val_new))
            yield idx_tr_new, idx_val_new
    else:
        idx_tr_new, idx_val_new = train_test_split(df, random_state=random_state, stratify=df[target], test_size=0.1)
        yield idx_tr_new, idx_val_new
        
        

class StackingModel(nn.Module):
        def __init__(self, models, num_class = 10):
            super(StackingModel, self).__init__()
            self.models = models
            self.num_class = num_class
            self.post_model = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(3, 1)),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(8, 32, kernel_size=(3, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(),
                nn.Flatten(),
                nn.Linear(32*self.num_class, 8*self.num_class),
                nn.Linear(8*self.num_class, self.num_class,)
            )
        def forward(self, x):
            predictions = []
            with torch.no_grad():
                for model in self.models:
                    y = model(x)
                    predictions.append(y)
            predictions = torch.stack(predictions).permute(1, 0, 2).unsqueeze(1)#.cuda()
#             print(predictions.size())
            y = self.post_model(predictions)
            return y
        
def main(config):
#     opts = config()
#     path = opts.path
    path = "../input"
    train_df = pd.read_csv(f'{path}/train.csv')
    
    n_train = len(np.load("../input/ukiyoe-train-imgs.npz")["arr_0"])
    n_test = len(np.load("../input/ukiyoe-test-imgs.npz")["arr_0"])
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')
#     probabilities = np.zeros((n_test, opts.num_class))
    probabilities = np.zeros((n_test, 10))
    test_ids = np.array(list(range(n_test)))+1
    
#     for fold, (train_ids_new, valid_ids_new) in enumerate(stratified_groups_kfold(train_df, target='y', n_splits=opts.fold_max, random_state=0)):
    for fold, (train_ids_new, valid_ids_new) in enumerate(stratified_groups_kfold(train_df, target='y', n_splits=5, random_state=0)):
        train_ids_new.to_csv(f'csvs/train_fold{fold}.csv')
        valid_ids_new.to_csv(f'csvs/valid_fold{fold}.csv')
        train_ids_new = train_ids_new.index.values
        valid_ids_new = valid_ids_new.index.values
        
        DEVICE = 'cuda'

        ACTIVATION = None
        
        stacked_model_paths = ["../logs/JPUNet_se_resnext50_32x4d_diffrgrad", "../logs/JPUNet_densenet121_diffrgrad", "../logs/JPUNet_inceptionresnetv2_diffrgrad", "../logs/JPUNet_cbam_resnext50_32x4d_diffrgrad", "../logs/JPUNet_resnet50_diffrgrad",]
        stacked_model_encoders = ["se_resnext50_32x4d", "densenet121", "inceptionresnetv2", "cbam_resnext50_32x4d", "resnet50"]
        stacked_models = []
        for model_path, model_encoder in zip(stacked_model_paths, stacked_model_encoders):
            model_type = model_path.split("/")[-1].split("_")[0]
            stacked_model = get_model(model_type = model_type ,
                          encoder= model_encoder,
                          encoder_weights = None,
                          metric_branch = False,
                          middle_activation = "Mish",
                          last_activation = None,
                          num_classes = 10
#                           num_classes = opts.num_class
                         )
            stacked_model = sbn_convert_model(stacked_model)
            checkpoint = torch.load(f"{model_path}/fold{fold}/checkpoints/best.pth")
            stacked_model.load_state_dict(checkpoint["model_state_dict"])
            stacked_model.cuda()
            stacked_models.append(stacked_model)
        stacking_model = StackingModel(stacked_models)
        
        
        train_dataset = UkiyoeDataset(datatype='train', img_ids=train_ids_new, transforms = get_training_augmentation((256, 256)), preprocessing=get_preprocessing(None))
        valid_dataset = UkiyoeDataset(datatype='train', img_ids=valid_ids_new, transforms = get_validation_augmentation((256, 256)), preprocessing=get_preprocessing(None))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True, worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True, worker_init_fn=worker_init_fn)

        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }

        num_epochs = 50
        logdir = f"../log/stacking/fold{fold}" 
        optimizer = get_optimizer(optimizer="RAdam", lookahead=False, model=stacking_model, separate_decoder=False, lr=1e-3)
        opt_level = 'O1'
        stacking_model.cuda()
#         model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level) # not working using adacos
        print(stacking_model)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        runner = SupervisedRunner()
        callbacks=[LINENotifyCallBack(), AccuracyCallback(activation="Softmax")]
        callbacks.append(CutMixCallback(alpha=0.25))
#         callbacks.append(EarlyStoppingCallback(patience=5, min_delta=0.001))
#         if opts.accumeration is not None:
#             callbacks.append(CriterionCallback())
#             callbacks.append(OptimizerCallback(accumulation_steps=opts.accumeration))
        print(f"############################## Start training of fold{fold}! ##############################")
        runner.train(
            model=stacking_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            num_epochs=num_epochs,
            main_metric = 'accuracy01', 
            minimize_metric = False,
            verbose=True
        )
        print(f"############################## Finish training of fold{fold}! ##############################")
        
        checkpoint = torch.load(f"{logdir}/checkpoints/best.pth")
        stacking_model.load_state_dict(checkpoint["model_state_dict"])
#         if opts.tta:
        stacking_model = tta.TTAWrapper(stacking_model, tta.fliplr_image2label)
#             tta_model = tta.TTAWrapper(model, tta.tencrop_image2label, crop_size=(int(opts.img_size[0]*0.9), int(opts.img_size[1]*0.9)))
        stacking_model = tta.TTAWrapper(stacking_model, bright_hl_image2label)
#         else:
#             pass
        TEST_DEVICE="cuda:0"
        stacking_model.to(TEST_DEVICE)
        
        count=0
        
#         test_dataset =  UkiyoeDataset(datatype='test', img_ids=test_ids, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        test_dataset =  UkiyoeDataset(datatype='test', img_ids=test_ids, transforms = get_validation_augmentation((256, 256)), preprocessing=get_preprocessing(None))
#         test_loader = DataLoader(test_dataset, batch_size=opts.batchsize, shuffle=False, num_workers=opts.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
        for i, (batch) in enumerate(tqdm.tqdm(test_loader, desc="batch loop", leave=False)):
            with torch.no_grad():
                probability_batch = stacking_model(batch.to(TEST_DEVICE))
#             print(probability_batch.cpu().numpy().shape)
            probabilities[count:count+batch.size(0)] += softmax(probability_batch.cpu().numpy())
            count += batch.size(0)
        del loaders
        del runner
        torch.cuda.empty_cache()
        gc.collect()
#     probabilities /= opts.fold_max
    probabilities /= 5
#     if opts.tta:
    np.save(f'probabilities/stacking_tta_test.npy', probabilities)
#     else:
#         np.save(f'probabilities/stacking_test.npy', probabilities)
    labels = probabilities.argmax(axis=1)
    torch.cuda.empty_cache()

    del probabilities
    gc.collect()
    
    ############# predict ###################
    df = pd.DataFrame(labels, columns=['y'])
    df.index.name = 'id'
    df.index = df.index + 1
#     if opts.tta:
    df.to_csv(f'predicts/stacking_tta.csv', float_format='%.5f')
#     else:
#         df.to_csv(f'predicts/stacking.csv', float_format='%.5f')

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
        
    
    
    
    