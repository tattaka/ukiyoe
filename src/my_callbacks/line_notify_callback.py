import requests
import os
from pathlib import Path

import numpy as np

import torch
from catalyst.dl.core import Callback, CallbackOrder
from catalyst.dl.callbacks.checkpoint import CheckpointCallback
from catalyst.dl.core.state import RunnerState

class LINENotifyCallBack(CheckpointCallback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """
    def __init__(
        self,
        save_n_best: int = 1,
        resume: str = None,
        resume_dir: str = None,
        metric_filename: str = "_metrics.json"
    ):
        """
        Args:
            save_n_best (int): number of best checkpoint to keep
            resume (str): path to checkpoint to load
                and initialize runner state
            metric_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(save_n_best, resume, resume_dir, metric_filename)

    def on_stage_end(self, state: RunnerState):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:3.4f}".format(
                    filepath=filepath, metric=checkpoint_metric
                ) for filepath, checkpoint_metric, _ in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)
        self.send_line_notification("Top best models:" + top_best_metrics_str)
        
    def send_line_notification(self, message):
        line_token = os.environ['LINE_TOKEN']
        endpoint = 'https://notify-api.line.me/api/notify'
        message = "\n{}".format(message)
        payload = {'message': message}
        headers = {'Authorization': 'Bearer {}'.format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)
        
        
__all__ = ["LINENotifyCallBack"]