import torch.nn as nn
import torchvision
import torch
from typing import Optional, Type

from .models import *
from .commons import *

def get_model(model_type: str = 'SimpleNet', # or "ACPNet" or "JPUNet" 
              encoder: str = 'resnet18',
              encoder_weights: str = 'imagenet',
              metric_branch:bool = False,
              middle_activation: str = "ReLU", # or "Swish" or "Mish"
              last_activation:str = "Softmax", # or "Sigmoid" or None
              num_classes: int = 10, 
              tta:bool = False,
             ):
    
    if model_type == 'SimpleNet':
        model = SimpleNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            metric_branch = metric_branch,
            num_classes=num_classes,
            last_activation=last_activation,
            tta=tta, 
        )
    elif model_type == "ACPNet":
        model = ACPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            metric_branch = metric_branch,
            num_classes=num_classes,
            last_activation=last_activation,
            tta=tta, 
        )
    elif model_type == "JPUNet":
        model = JPUNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            metric_branch = metric_branch,
            num_classes=num_classes,
            last_activation=last_activation,
            tta=tta, 
        )
    else:
        model = None
    if middle_activation == "Swish":
        model = convert_model_ReLU2Swish(model)
    elif middle_activation == "Mish":
        model = convert_model_ReLU2Mish(model)
    return model

def convert_model_ReLU2Swish(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model_ReLU2Swish(mod)
        mod = DataParallelWithCallback(mod)
        return mod
    
    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = Swish()
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2Swish(child))
    return mod

def convert_model_ReLU2Mish(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model_ReLU2Mish(mod)
        mod = DataParallelWithCallback(mod)
        return mod
    
    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = Mish()
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2Mish(child))
    return mod