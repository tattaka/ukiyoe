import sys
from functools import partial
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from utils import losses
class Config(object):
    def __init__(self):
        
        self.path = "../input"
        self.model_type = "JPUNet"
        self.fold_max = 5
        self.encoder = "densenet121"
        self.encoder_weights = None
        self.mid_activation = "Mish"
        self.metric_branch = False
        self.pretrain_model_type = None
        self.pretrain_path = None
        self.logdir = "../logs/"+self.model_type+"_"+self.encoder+"_diffrgrad"

        self.img_size = (256, 256)
        
        self.batchsize = 32
        self.num_class = 10
        
        self.num_workers = 8
        self.max_epoch = 150
        
        self.optimizer = "diffRGrad"
        self.lr = 1e-3
        self.lr_e = 1e-3
        self.lookahead = False
        self.tta = False
        self.early_stop = False
        self.mixup = True
        
        self.scheduler = partial(CosineAnnealingLR, T_max=self.max_epoch, eta_min=1e-8)
        self.criterion = losses.SmoothCrossEntropyLoss(smoothing=0.1)
        self.accumeration = None
        
        
        