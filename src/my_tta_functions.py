import numpy as np
from torch import Tensor, nn


def bright_hl_image2label(model: nn.Module, image: Tensor, b_value:float = 0.05) -> Tensor:
    output = model(image) + model(brightness(image, b_value)) + model(brightness(image, -b_value))
    one_over_3 = float(1.0 / 3.0)
    return output * one_over_3

def togray_image2label(model: nn.Module, image: Tensor) -> Tensor:
    output = model(image) + model(togray(image))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2

def brightness(image: Tensor, beta:float, beta_by_max:bool=True) -> Tensor:
    dtype = image.dtype
    if beta_by_max:
        max_value = 1.0
        image += beta * max_value
    else:
        image += beta * torch.mean(image)
    return image

def togray(image: Tensor) -> Tensor: #BGR->GlayScale
    gray = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    image[0] = gray
    image[1] = gray
    image[2] = gray
    return image