import gc
import re

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.image_net import image_net
from models.batch_normalization import BatchNorm2d
from models.resnetQuan import Conv2dQuan, LinearQuan
from models.weight_network import weight_network

def get_model_conv_weights(model: nn.Module) -> dict[str, Tensor]:
    weights_dict = {}
    for k, v in model.named_parameters():
        if re.search('layer.*conv.*', k) != None:
            weights_dict[k] = v.clone().detach()
    return weights_dict

def get_all_module_with_weight_net(model: nn.Module) -> set[nn.Module]:
    all_modules_with_weight_network = []
    for parent_module in model.modules():
        if hasattr(parent_module, 'weight_net'):
            all_modules_with_weight_network.append(parent_module)

    return all_modules_with_weight_network

def get_all_activation_modules(model: nn.Module) -> set[nn.Module]: # return all module before a convolution or fc layer, which are of type MaxPool2dQuan, AdaptiveAvgPool2dQuan, ReLUQuan
    all_activation_modules = []
    for module in model.modules():
        if type(module) in [Conv2dQuan, LinearQuan]:
            all_activation_modules.append(module)

    return all_activation_modules

def reconstruction_loss(weight_fp: Tensor, recons_weight: Tensor):
    if len(weight_fp.shape) == 2:
        return torch.mean(torch.clamp(torch.abs(weight_fp - recons_weight), min=1e-12))
    else:
        return torch.mean(torch.sqrt(torch.clamp(torch.sum((weight_fp - recons_weight) ** 2, dim=(2, 3)), min=1e-12)))

def activation_reconstruction_loss(weight_fp: Tensor, recons_weight: Tensor):
    if len(weight_fp.shape) == 2:
        return torch.mean(torch.clamp(torch.abs(weight_fp - recons_weight), min=1e-12))
    else:
        return torch.mean(torch.sqrt(torch.clamp(torch.sum((weight_fp - recons_weight) ** 2, dim=(1, 2, 3)), min=1e-12)))

def twn_reconstructed_weights(weight_fp: Tensor):
    weight_fp_org = weight_fp
    if len(weight_fp.shape) == 2:
        weight_fp = weight_fp[:, :, None, None]
    delta = 0.75 * torch.sum(weight_fp.abs(), dim=(1, 2, 3)) / (weight_fp[0].nelement())
    absw = torch.abs(weight_fp)
    Iw = absw > delta[:, None, None, None]
    alpha2 = (1 / torch.sum(Iw, dim=(1, 2, 3))) * (torch.sum(absw * Iw, dim=(1, 2, 3)))
    w_ = 1 * (weight_fp > delta[:, None, None, None]) + (-1) * (weight_fp < -delta[:, None, None, None])
    recons_w = alpha2[:, None, None, None] * w_
    if len(weight_fp_org.shape) == 2:
        recons_w = recons_w[:, :, 0, 0]
    return recons_w

def compare_twn_wn_reconstruction(model: nn.Module):
    all_modules_with_weight_network = get_all_module_with_weight_net(model)
    twn_recon_loss = []
    wn_recon_loss_eps = []
    wn_recon_loss = []
    for module in all_modules_with_weight_network:
        weight_net = module.weight_net
        weight_net.eval()
        weight = module.weight

        with torch.no_grad():
            recons_w = weight_net.get_reconstructed_weights(weight, module.epsilon)
            wn_recon_loss_eps.append(reconstruction_loss(weight, recons_w))

            recons_w = weight_net.get_reconstructed_weights(weight, 0)
            wn_recon_loss.append(reconstruction_loss(weight, recons_w))

            recons_w = twn_reconstructed_weights(weight)
            twn_recon_loss.append(reconstruction_loss(weight, recons_w))

    twn_recon_loss = torch.mean(torch.tensor(twn_recon_loss))
    wn_recon_loss_eps = torch.mean(torch.tensor(wn_recon_loss_eps))
    wn_recon_loss = torch.mean(torch.tensor(wn_recon_loss))

    return {'wn_recon_loss_eps': wn_recon_loss_eps.item(), 'wn_recon_loss': wn_recon_loss.item(), 'twn_recon_loss': twn_recon_loss.item()}

def quantize_fp_model_weights(model_fp: nn.Module, weight_net: weight_network):
    model_fp.requires_grad_(False)
    for k, v in model_fp.named_parameters():
        if re.search('layer.*conv.*', k) != None:
            v[:] = weight_net.get_reconstructed_weights(weight_fp=v, epsilon=0)

def plot_activations(model):
    dataset = image_net('train')
    dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=20, shuffle=True)

    model.eval()
    progress_bar = tqdm(dataloader)
    activation_nets = get_all_activation_modules(model)

    for activation_net in activation_nets:
        activation_net.save_activations = True
        activation_net.train()

    for xs, ys in progress_bar:
        with torch.no_grad():
            model(xs.cuda())

            # loss = []
            # for activation_net in activation_nets:
            #     loss.append(activation_reconstruction_loss(activation_net.last_fp_activation, activation_net.last_quan_activation))
            #
            # loss = torch.mean(torch.stack(loss))
            # progress_bar.set_description(f"loss: {loss.item(): .6f}")

        if progress_bar.n > 500:
            for activation_net in activation_nets:
                activation_net.last_fp_activation = None
                activation_net.last_quan_activation = None
                activation_net.save_activations = False
            break

        # if progress_bar.n % 500 == 0:
        #     with torch.no_grad():
        #         activation_rows = 3
        #         qactivation_rows = 3
        #         cols = 6
        #         fig, ax = plt.subplots(1 + activation_rows + qactivation_rows, cols)
        #         for c in range(cols):
        #             img = xs[c].cpu().numpy().transpose((1, 2, 0))
        #             img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #             ax[0, c].imshow(img)
        #         for a in range(activation_rows):
        #             for c in range(cols):
        #                 img = (activation_nets[0].last_fp_activation)[c][a].cpu().numpy()[:, :, None]
        #                 img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #                 ax[1+a, c].imshow(img)
        #         for a in range(qactivation_rows):
        #             for c in range(cols):
        #                 qact = activation_nets[0].last_quan_activation
        #                 img = (qact)[c][a].detach().cpu().numpy()[:, :, None]
        #                 img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #                 ax[1 + activation_rows + a, c].imshow(img)
        #         plt.show()
        #         torch.save(model, 'act_quan.model')
        #         torch.cuda.empty_cache()
        #         gc.collect()

