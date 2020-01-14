import numpy as np
import torch
from cls_models import get_model

if __name__ == '__main__':
    x = np.zeros((3, 3, 384, 576), dtype="f")
    x = torch.from_numpy(x)
    print("input shape:", x.size())
    model = get_model(
        model_type = 'SimpleNet',
        encoder= 'resnet18',
        encoder_weights= 'imagenet',
        metric_branch = True,
        metric_weight = 0.,
        middle_activation = "Swish",
        last_activation = "LogSoftmax",
        num_classes = 10)
    print(model)
    y = model(x)
    print(y.shape)
    target = torch.tensor([1, 0, 4])
    loss = torch.nn.NLLLoss()
    print(loss(y, target))
    print("out shape:", y.size())