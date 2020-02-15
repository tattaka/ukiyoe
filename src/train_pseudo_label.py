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
# from antialiased_cnns_converter import convert_model as aacnn_convert_model

from my_callbacks import CutMixCallback, LINENotifyCallBack, PixMixCallback, CutMixAndMixupCallback


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
torch.backends.cudnn.deterministic = True
np.random.seed(666)
torch.manual_seed(666)
# torch.backends.cudnn.benchmark = True

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
        
    
def main(config):
    opts = config()
    path = opts.path
    train_df = pd.read_csv(f'{path}/train.csv')
    test_df = pd.read_csv('predicts/stacking.csv')
    pl_prob = np.load('probabilities/stacking_test.npy')
    
    n_train = len(np.load("../input/ukiyoe-train-imgs.npz")["arr_0"])
    n_test = len(np.load("../input/ukiyoe-test-imgs.npz")["arr_0"])
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')
    pseudo_label_idx = []
    pl_max = pl_prob.max(axis=1)
    for i, pl_max_i in enumerate(pl_max):
        if pl_max_i > 0.8:
            pseudo_label_idx.append(i)
    pseudo_label_df = test_df.iloc[pseudo_label_idx]
        
    for fold, ((train_ids_new, valid_ids_new), (train_pl_ids_new, valid_pl_ids_new)) in enumerate(zip(stratified_groups_kfold(train_df, target='y', n_splits=opts.fold_max, random_state=0), stratified_groups_kfold(pseudo_label_df, target='y', n_splits=opts.fold_max, random_state=0))):
        train_ids_new.to_csv(f'csvs/train_fold{fold}.csv')
        valid_ids_new.to_csv(f'csvs/valid_fold{fold}.csv')
        train_ids_new = train_ids_new.index.values
        valid_ids_new = valid_ids_new.index.values
        train_pl_ids_new.to_csv(f'csvs/train_pl_fold{fold}.csv')
        valid_pl_ids_new.to_csv(f'csvs/valid_pl_fold{fold}.csv')
        train_pl_ids_new = train_pl_ids_new.index.values
        valid_pl_ids_new = valid_pl_ids_new.index.values
        
        DEVICE = 'cuda'

        ACTIVATION = None
        model = get_model(model_type = opts.model_type,
                          encoder= opts.encoder,
                          encoder_weights = opts.encoder_weights,
                          metric_branch = opts.metric_branch,
                          middle_activation = opts.mid_activation,
                          last_activation = None,
                          num_classes = opts.num_class
                         )
        model = sbn_convert_model(model)
        
#         preprocessing_fn = encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        if opts.pretrain_path:
            import copy
            pretrain_model = get_model(model_type = opts.pretrain_model_type,
                                       encoder= opts.encoder,
                                       encoder_weights = opts.encoder_weights,
                                       metric_branch = False,
                                       middle_activation = opts.mid_activation,
                                       last_activation = None,
                                       num_classes = opts.num_class
                                      )
            pretrain_model = sbn_convert_model(pretrain_model)
            checkpoint = torch.load(opts.pretrain_path+f"/fold{fold}/checkpoints/best_full.pth")
            pretrain_model.load_state_dict(checkpoint['model_state_dict'])
#             model.load_state_dict(checkpoint['model_state_dict'])
            if opts.pretrain_model_type == "SimpleNet":
                model.encoder = copy.deepcopy(pretrain_model.encoder)
            else:
                model = copy.deepcopy(pretrain_model)
            del pretrain_model
        
        def remove_dropout(module):
            mod = module
            if isinstance(module, nn.Dropout):
                mod = torch.nn.Identity()
            for name, child in module.named_children():
                mod.add_module(name, remove_dropout(child))
            return mod
        model = remove_dropout(model)
        
        train_dataset = UkiyoeDataset(datatype='train', img_ids=train_ids_new, transforms = get_training_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        valid_dataset = UkiyoeDataset(datatype='valid', img_ids=valid_ids_new, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        
        train_pl_dataset = UkiyoePseudoLabelDataset(pseudo_label_df=test_df,img_ids=train_pl_ids_new, transforms = get_training_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        valid_pl_dataset = UkiyoePseudoLabelDataset(pseudo_label_df=test_df, img_ids=valid_pl_ids_new, transforms = get_validation_augmentation(opts.img_size), preprocessing=get_preprocessing(None))
        
        train_dataset = ConcatDataset([train_dataset, train_pl_dataset])
        valid_dataset = ConcatDataset([valid_dataset, valid_pl_dataset])
        
        train_loader = DataLoader(train_dataset, batch_size=opts.batchsize, shuffle=True, num_workers=opts.num_workers, drop_last=True, worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=opts.batchsize, shuffle=False, num_workers=opts.num_workers, drop_last=True, worker_init_fn=worker_init_fn)

        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
        num_epochs = opts.max_epoch
        logdir = f"{opts.logdir}/fold{fold}" 
        optimizer = get_optimizer(optimizer=opts.optimizer, lookahead=opts.lookahead, model=model, separate_decoder=True, lr=opts.lr, lr_e=opts.lr_e)
        opt_level = 'O1'
        model.cuda()
#         model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level) # not working using adacos
        print(model)
        scheduler = opts.scheduler(optimizer)
        criterion = opts.criterion
        runner = SupervisedRunner()
        callbacks=[LINENotifyCallBack(), AccuracyCallback(activation="Softmax")] #WIP:Add LINE NotifyCallBack
        if opts.early_stop:
            callbacks.append(EarlyStoppingCallback(patience=10, min_delta=0.001))
        if opts.mixup:
#             callbacks.append(MixupCallback(alpha=0.25))
#             callbacks.append(CutMixCallback(alpha=0.25))
            callbacks.append(CutMixAndMixupCallback(alpha=0.8))
#             callbacks.append(PixMixCallback(alpha=0.25))
        if opts.accumeration is not None:
            callbacks.append(CriterionCallback())
            callbacks.append(OptimizerCallback(accumulation_steps=opts.accumeration))
        print(f"############################## Start training of fold{fold}! ##############################")
        runner.train(
            model=model,
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
        del model
        del loaders
        del runner
        torch.cuda.empty_cache()
        gc.collect()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default_cfg')
    args = parser.parse_args()
    config = import_module("configs."+ args.config)
    main(config.Config)
        
    
    
    
    