import torch
from torch import nn

from models.batch_normalization import BatchNorm2d


class activation_network(nn.Module):

    def __init__(self, channels, height, width):
        super().__init__()
        self.b1 = BatchNorm2d(num_features=channels)
        self.epsilon = 5e-2

        self.save_activations = False
        self.last_fp_activation = None
        self.last_quan_activation = None

    def forward(self, x):
        _x = x
        if len(x.shape) == 2:
            _x = x[:, :, None, None]

        y = self.b1(_x)  # this is equilvatent to (x - dict(self.b1.named_buffers())['running_mean'][None, :, None, None]) / torch.sqrt(dict(self.b1.named_buffers())['running_var'] + 1e-5)[None, :, None, None], self.b1(x)
        if self.save_activations:
            self.last_fp_activation = y * torch.sqrt(dict(self.b1.named_buffers())['running_var'])[None, :, None, None]

        # y = y / (3 * torch.sqrt(dict(self.b1.named_buffers())['running_var']))[None, :, None, None]
        # y = self.get_reconstructed_weights(y, self.epsilon)

        if self.save_activations:
            self.last_quan_activation = y

        if len(x.shape) == 2:
            y = y[:, :, 0, 0]

        return y

    def q(self, x, epsilon):
        _x = 1.5 * x
        v1 = _x * epsilon
        v2 = (_x + 0.5) * (epsilon) + (-1 + epsilon / 2)
        v3 = (_x - 0.5) * (epsilon) + (1 - epsilon / 2)
        return torch.logical_and(_x >= -0.5, _x <= 0.5) * v1 + (_x < -0.5) * v2 + (_x > 0.5) * v3

    def get_reconstructed_weights(self, w_b, epsilon):
        w_q = self.q(w_b, epsilon)
        recons_w = ((3 * torch.sqrt(dict(self.b1.named_buffers())['running_var'])) / (2 + epsilon))[None, :, None, None] * w_q

        return recons_w