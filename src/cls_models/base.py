import torch
import numpy as np
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class BaseModel(Model):
    def __init__(self, encoder, cls_head, tta=False):
        super().__init__()
        self.encoder = encoder
        self.cls_head = cls_head
        self.tta = tta

    def forward(self, x):
        if self.tta: 
            clop_shape = (int(x.shape[-2]*0.8), int(x.shape[-1]*0.8))
            from pytorch_toolbelt.inference import tta
            model = lambda x: self.cls_head(self.encoder(x))
            model = tta.TTAWrapper(model, tta.fivecrop_image2label, crop_size=clop_shape)
            y = model(x)
        else:
            y = self.cls_head(self.encoder(x))
        return y

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
        return x