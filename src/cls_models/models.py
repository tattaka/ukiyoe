import numpy as np
import torch
from torch import nn
from .base import BaseModel
from .encoders import get_encoder
from .heads import SimpleNetHead, ACPNetHead, JPUNetHead

class SimpleNet(BaseModel):
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 metric_branch = False,
                 last_activation:str = "LogSoftmax",
                 num_classes=10,
                 tta=False, ):
        
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )
        cls_head = SimpleNetHead(
            encoder_channels=encoder.out_shapes,
            metric_branch=metric_branch,
            last_activation=last_activation,
            num_classes=num_classes,
        )
        super().__init__(encoder, cls_head, tta)
        
        self.name = 'simplenet-{}'.format(encoder_name)
        
class ACPNet(BaseModel):
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 metric_branch = False,
                 last_activation:str = "LogSoftmax",
                 num_classes=10,
                 tta=False, ):
        
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )
        cls_head = ACPNetHead(
            encoder_channels=encoder.out_shapes,
            metric_branch=metric_branch,
            last_activation=last_activation,
            num_classes=num_classes,
        )
        super().__init__(encoder, cls_head, tta)
        
        self.name = 'acpnet-{}'.format(encoder_name)

class JPUNet(BaseModel):
    def __init__(self,
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 metric_branch = False,
                 last_activation:str = "LogSoftmax",
                 num_classes=10,
                 tta=False, ):
        
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )
        cls_head = JPUNetHead(
            encoder_channels=encoder.out_shapes,
            metric_branch=metric_branch,
            last_activation=last_activation,
            num_classes=num_classes,
        )
        super().__init__(encoder, cls_head, tta)
        
        self.name = 'jpunet-{}'.format(encoder_name)