import logging
import re
from math import floor
from pathlib import Path

import jsons
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from dataset.image_net import image_net
from evaluation.utils import evaluate_imagenet
from models.resnetQuan import Conv2dQuan, LinearQuan, ResNetQuan
from utils.utils import compare_twn_wn_reconstruction


class epoch_result:
    def __init__(self, epoch):
        self.epoch = epoch
        self.val_loss_accuracy = []
        self.before_training_wn_loss = None
        self.after_training_wn_loss = None

def train_model(model: nn.Module, lr: list[float], weight_lambda: list[float], activation_lambda: list[float], train_output_path: Path, lr_scaling: list[bool], grad_scaling: list[bool]):
    epochs = len(lr)
    dataset = image_net('train')
    dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=20, shuffle=True)

    param_groups = []
    param_group_names = []
    for name, parameter in model.named_parameters():
        param_groups.append({'params': [parameter]})
        param_group_names.append(name)
    optimizer = torch.optim.SGD(param_groups, lr=1e-2, momentum=0.9)

    train_output_path.mkdir(parents=True, exist_ok=True)
    epoch_percentage_val_evaluation = 0.2
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(str(Path.joinpath(train_output_path, 'log.log')), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s    |   %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    ce_loss = CrossEntropyLoss()
    kl_loss = KLDivLoss(reduction="batchmean")
    for epoch in range(1, epochs+1):
        index = epoch-1
        progress_bar = tqdm(dataloader)
        progress_bar_val_step = floor(progress_bar.total * 0.2)
        epoch_res = epoch_result(epoch)
        model.train()

        # for p in list(model.named_parameters()):
        #     if p[0].endswith("qscale"):
        #         p[1].requires_grad_(False)

        modules = model.named_modules()
        for name, module in modules:
            if type(module) == Conv2dQuan and module.kernel_size[0] != 1 and module.kernel_size[0] != 7:
                module.weight_quantization_module.epsilon = weight_lambda[index]
                module.weight_quantization_module.grad_scaling = grad_scaling[index]
                print(f"set weight_lambda to : {module.weight_quantization_module.epsilon}, for {name}, {module.weight_quantization_module.grad_scaling}")
                if hasattr(module, 'activation_quantization_module'):
                    module.activation_quantization_module.epsilon = activation_lambda[index]
                    if not re.search('.*layer.*conv2.*', name) != None:
                        module.activation_quantization_module.grad_scaling = grad_scaling[index]
                    print(f"set activation_lambda to : {module.activation_quantization_module.epsilon}, for {name}, {module.activation_quantization_module.grad_scaling}")

        for n, g in zip(param_group_names, optimizer.param_groups):
            if lr_scaling[index]:
                if re.search('.*layer.*conv.*', n) != None:
                    g['lr'] = lr[index]
                    print(f"set lr to : {g['lr']} for {n} ")
                else:
                    g['lr'] = lr[index]
                    print(f"set lr to : {g['lr']} for {n} ")
            else:
                g['lr'] = lr[index]
                print(f"set lr to : {g['lr']} for {n} ")

            module.lr = lr[index]

        for xs, ys in progress_bar:
            # if epoch > 10:
            xs = xs.cuda()
            # with torch.no_grad():
            #     prob_pretrained = torch.softmax(pretrained(xs), dim=1)
            # prob = torch.log_softmax(model(xs), dim=1)
            #
            # loss = kl_loss(prob, prob_pretrained)
            # else:
            logit = model(xs)
            loss = ce_loss(logit, ys.cuda())

            optimizer.zero_grad()
            loss.backward()
            # for p in list(model.named_parameters()):
            #     if p[0].endswith("qscale"):
            #         p[1].grad *= 1e3
            optimizer.step()

            progress_msg = f'epoch: {epoch}, step: {progress_bar.n}, train_batch_loss: {loss.item() : .8f}, val_loss_accuracy: {epoch_res.val_loss_accuracy}'
            progress_bar.set_description(progress_msg)
            logger.debug(progress_msg)

            if (progress_bar.n+1) % progress_bar_val_step == 0:
                val_eval = evaluate_imagenet(model)
                epoch_res.val_loss_accuracy.append(((progress_bar.n+1) / progress_bar_val_step * epoch_percentage_val_evaluation, val_eval))
                model.lr = optimizer.__getstate__()['param_groups'][0]['lr']
                torch.save(model, str(Path.joinpath(train_output_path, f'model_epoch_{epoch}.model')))

                logger.info(f'epoch: {epoch}, step: {progress_bar.n}, before: {val_eval}')
                model.train()


        # for p in list(model.named_parameters()):
        #     if p[0].endswith("qscale"):
        #         p[1].requires_grad_(False)

        # if epoch == 7 or epoch == 12:


        # if epoch == 9:
        #     scheduler.step()

        logger.info('---------------------------------------------------------------')
        logger.info(jsons.dumps(epoch_res))
        # logger.info(compare_twn_wn_reconstruction(model))
        logger.info('---------------------------------------------------------------')