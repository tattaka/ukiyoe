import requests
import os

from typing import List  # isort:skip

import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder
from catalyst.dl.callbacks import CriterionCallback, OptimizerCallback
from catalyst.dl.core.state import RunnerState
from pathlib import Path
from .cutmix_callback import *

class MeanTeacherOptimizerCallback(OptimizerCallback):
    def __init__(self, ema_model, **kwargs):
        super().__init__(**kwargs)
        self.ema_model = ema_model
    
    def on_stage_start(self, state: RunnerState):
        """On stage start event"""
        super().on_stage_start(state)
        state.checkpoint_data["ema_model"] = self.ema_model.state_dict()
        
    def on_batch_end(self, state):
        """On batch end event"""
        super().on_batch_end(state)
        if state.epoch > 20:
            alpha_ema = 0.999
        elif state.epoch > 10:
            alpha_ema = 0.99
        elif state.epoch > 1:
            alpha_ema = 0.9
        else:
            alpha_ema = 0.5
        
        self.ema_model.load_state_dict(state.checkpoint_data["ema_model"])
        self.ema_model = update_ema_variables(state.model, self.ema_model, alpha_ema, state.step)
        state.checkpoint_data["ema_model"] = self.ema_model.state_dict()
        

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_model

        

class MeanTeacherCallback(CutMixCallback):
    """
    Callback to do MeanTeacher.
    Paper: https://github.com/CuriousAI/mean-teache
    Note:
        MeanTeacher callback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
        self,
        ema_model,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_model = ema_model
        self.consistency = 100.0
        self.consistency_rp = 5
    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)
        classification_loss = super()._compute_loss(state, criterion)
        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        ema_logits = {}
        for f in self.fields:
            self.ema_model.load_state_dict(state.checkpoint_data["ema_model"])
            ema_logits[self.output_key] = self.ema_model(ema_augment(state.input[f]))
        consistency_weight = get_current_consistency_weight(state.epoch, self.consistency, self.consistency_rp)
        ema_logit = Variable(ema_logits[self.output_key].detach().data, requires_grad=False)
        consistency_loss = consistency_weight * softmax_mse_loss(pred, ema_logits[self.output_key]) / ema_logits[self.output_key].size(0)
        return loss + consistency_loss*0.1

def ema_augment(images):
    var_limit=(0.07, 0.2)
    for i, image in enumerate(images):
        var = random.uniform(var_limit[0], var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        gauss = random_state.normal(0, sigma, (image.size(-2), image.size(-1)))
        gauss = torch.from_numpy(gauss.astype('f')).to('cuda')
        image += gauss
    return images


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def get_current_consistency_weight(epoch, consistency, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))