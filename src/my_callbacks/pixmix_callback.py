import requests
import os

from typing import List  # isort:skip

import numpy as np

import torch
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder
from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.core.state import RunnerState
from pathlib import Path

class PixMixCallback(CriterionCallback):
    """
    Callback to do PixMix augmentation.
    Paper: https://arxiv.org/abs/
    Note:
        PixMixCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
        self,
        fields: List[str] = ("features", ),
        pix_size:int = 16,
#         pix_size:int = 1,
        alpha=1.0,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.pix_size = pix_size
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            w = int(state.input[f].size(-1) / self.pix_size)
            h = int(state.input[f].size(-2) / self.pix_size)
            idx = np.arange(w * h)
            idx = np.random.choice(idx, int(self.lam * w * h), replace=False)
            tmp = np.zeros(w * h)
            tmp[idx] = 1
            tmp = tmp.reshape(1, 1, h, w)
            tmp = torch.from_numpy(tmp).type(torch.FloatTensor).to(state.input[f].device)
            if self.pix_size > 1:
                tmp = F.interpolate(tmp, size=(state.input[f].size(-2), state.input[f].size(-1)))
            self.lam = tmp.sum() / (state.input[f].size(-1)*state.input[f].size(-2))
            state.input[f] = state.input[f] * tmp + state.input[f][self.index] * (1 - tmp )

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss

__all__ = ["PixMixCallback"]