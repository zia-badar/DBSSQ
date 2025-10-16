import gc
import re
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
from torch import nn
from torchvision.models import ResNet18_Weights

from evaluation.utils import evaluate_imagenet
from models.batch_normalization import BatchNorm2d
from models.resnetQuan import Conv2dQuan, LinearQuan, ResNetQuan, quantization
from models.resnetQuan import resnet18Quan
from train.train import train_model
from utils.utils import plot_activations

# def _backward_hook(module, grad_input, grad_output):
#     if re.search('.*activation.*', module.name) != None:
#         # cal = grad_input[0]
#         if not re.search('.*layer.*conv2.*', name) != None:
#             cal = grad_input[0] / (module.epsilon)
#         else:
#             cal = grad_input[0]
#     elif re.search('.*weight.*', module.name) != None:
#         cal = grad_input[0]
#     # cal2 = grad_input[0] * (-torch.log2(torch.tensor(module.epsilon)))
#     # cal3 = grad_input[0] * 2**7
#     print(f"backward {module.name}, {len(grad_input)} {torch.mean(cal).item()}")
#     # if module.epsilon == 1:
#     #     return (grad_input[0])
#     # else:
#     #     return (cal3,)
#     return (cal,)


if __name__ == '__main__':
    # model = torch.load('iterations/iter_62/model_epoch_15.model', weights_only=False).cuda()
    # # # # #
    # modules = model.named_modules()
    # for name, module in modules:
    #     if type(module) == Conv2dQuan and module.kernel_size[0] != 1 and module.kernel_size[0] != 7:
    #         module.weight_quantization_module.epsilon = 0
    #         if hasattr(module, 'activation_quantization_module'):
    #             module.activation_quantization_module.epsilon = 0
    # print(evaluate_imagenet(model, show_tqdm=True))
    # exit(0)

    model = resnet18Quan(weights=ResNet18_Weights.IMAGENET1K_V1, use_activation_quantization=False).cuda()
    modules = model.modules()
    for module in modules:
        with torch.no_grad():
            if type(module) == Conv2dQuan:
                module.fixed_std = torch.std(module.weight.detach(), dim=(1, 2, 3))
    model = model.cuda()

    # named_modules = model.named_modules()
    # for name, module in named_modules:
    #     if type(module) == quantization:
    #         module.name = name + ' ' + module.name
    #         module.register_full_backward_hook(_backward_hook)

    train_model(model,
                lr=
                [1e-3,          1e-2, 1e-2, 1e-2,           1e-2, 1e-2, 1e-2,           1e-3, 1e-3, 1e-3,           1e-3, 1e-3,             1e-3,           1e-3,           1e-3],
                weight_lambda=
                [1,             2**-1, 2**-1, 2**-1,        2**-2, 2**-2, 2**-2,        2**-3, 2**-3, 2**-3,        2**-4, 2**-4,           2**-5,          2**-6,          2**-7],
                activation_lambda=
                [1,             2**-1, 2**-1, 2**-1,        2**-2, 2**-2, 2**-2,        2**-3, 2**-3, 2**-3,        2**-4, 2**-4,           2**-5,          2**-6,          2**-7],
                lr_scaling=
                [True,          True, True, True,           True, True, True,           True, True, True,           True, True,             True,           True,           True],
                grad_scaling=
                [True,          True, True, True,           True, True, True,           True, True, True,           True, True,             True,           True,           True],
                train_output_path=Path('iterations/iter_62/'))