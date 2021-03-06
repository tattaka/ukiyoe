import requests
import os

from typing import List  # isort:skip

import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder
from catalyst.dl.callbacks import CriterionCallback, OptimizerCallback, MixupCallback
from catalyst.dl.core.state import RunnerState
from pathlib import Path
from .cutmix_callback import *

class KnowledgeDistill():
    """
    Knowledge Distillaion support while fine-tuning the compressed model
    Geoffrey Hinton, Oriol Vinyals, Jeff Dean
    "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    """

    def __init__(self, teacher_model, kd_T=1):
        """
        Parameters
        ----------
        teacher_model : pytorch model
            the teacher_model for teaching the student model, it should be pretrained
        kd_T: float
            kd_T is the temperature parameter, when kd_T=1 we get the standard softmax function
            As kd_T grows, the probability distribution generated by the softmax function becomes softer
        """

        self.teacher_model = teacher_model
        self.kd_T = kd_T

    def _get_kd_loss(self, data, student_out, teacher_out_preprocess=None):
        """
        Parameters
        ----------
        data : torch.Tensor
            the input training data
        student_out: torch.Tensor
            output of the student network
        teacher_out_preprocess: function
            a function for pre-processing teacher_model's output
            e.g. when teacher_out_preprocess=lambda x:x[0]
            extract teacher_model's output (tensor1, tensor2)->tensor1
        Returns
        -------
        torch.Tensor
            weighted distillation loss
        """
        with torch.no_grad():
            kd_out = self.teacher_model(data)
        if teacher_out_preprocess is not None:
            kd_out = teacher_out_preprocess(kd_out)
        assert type(kd_out) is torch.Tensor
        assert type(student_out) is torch.Tensor
        assert kd_out.shape == student_out.shape
        soft_log_out = F.log_softmax(student_out / self.kd_T, dim=1)
        soft_t = F.softmax(kd_out / self.kd_T, dim=1)
        loss_kd = F.kl_div(soft_log_out, soft_t.detach(), reduction='batchmean')
        return loss_kd

    def loss(self, data, student_out):
        """
        Parameters
        ----------
        data : torch.Tensor
            Input of the student model
        student_out : torch.Tensor
            Output of the student model
        Returns
        -------
        torch.Tensor
            Weighted loss of student loss and distillation loss
        """
        return self._get_kd_loss(data, student_out)
    
    
    
class KnowledgeDistillCallback(CutMixCallback):
# class KnowledgeDistillCallback(MixupCallback):
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
        teacher_models,
        kd_T=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.knowledge_distill = []
        for t_model in teacher_models:
            self.knowledge_distill.append(KnowledgeDistill(t_model, kd_T))

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)
        pred = state.output[self.output_key]
        loss = super()._compute_loss(state, criterion)
        kd_losses = []
        for f in self.fields:
            for kd in self.knowledge_distill:
                kd_losses.append(kd.loss(state.input[f], pred))
        kd_loss = torch.stack(kd_losses).mean()
        return loss + kd_loss