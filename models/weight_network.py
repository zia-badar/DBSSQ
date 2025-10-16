import torch
from torch import nn


class weight_network(nn.Module):
    def __init__(self, kernel_height, kernel_width):
        super().__init__()
        layer_size = kernel_height * kernel_width
        self.returned_height = kernel_height
        self.returned_width = kernel_width

        self.a1 = nn.Flatten(start_dim=0, end_dim=1)
        self.a1_1 = nn.Flatten(start_dim=1, end_dim=2)
        self.a2 = nn.Linear(layer_size, out_features=layer_size, bias=True)
        self.a4 = nn.LeakyReLU()
        self.a5 = nn.Linear(layer_size, out_features=layer_size, bias=True)

        self.a9 = nn.Tanh()

        self.a11 = nn.Linear(layer_size, 1, bias=True)
        self.a13 = nn.Sigmoid()

    def forward(self, x, epsilon):
        _x = x
        if len(x.shape) == 2:
            _x = x[:, :, None, None]
        y = self.a1(_x)
        y = self.a1_1(y)
        y = self.a2(y)
        y4 = self.a4(y)
        y = self.a5(y4)
        o1 = self.a9(y)
        y = self.a11(y4)
        o2 = self.a13(y)

        if len(x.shape) == 2:
            return self.get_reconstructed_weights(x, o1.reshape(x.shape[0], x.shape[1]), torch.mean(o2.reshape(x.shape[0], -1), dim=1), epsilon)
        elif len(x.shape) == 4:
            return self.get_reconstructed_weights(x, o1.reshape(x.shape[0], x.shape[1], self.returned_height, self.returned_width), torch.mean(o2.reshape(x.shape[0], -1), dim=1), epsilon)

    def q(self, x, epsilon):
        _x = 1.5 * x
        v1 = _x * epsilon
        v2 = (_x + 0.5) * (epsilon) + (-1 + epsilon / 2)
        v3 = (_x - 0.5) * (epsilon) + (1 - epsilon / 2)
        return torch.logical_and(_x >= -0.5, _x <= 0.5) * v1 + (_x < -0.5) * v2 + (_x > 0.5) * v3

    def get_reconstructed_weights(self, weight_fp, w_b, alpha, epsilon):
        w_q = self.q(w_b, epsilon)
        if len(weight_fp.shape) == 2:
            with torch.no_grad():
                range_w = 4 * torch.std(weight_fp, dim=(1)).detach()
            recons_w = (alpha * range_w / (2 + epsilon))[:, None] * w_q
        elif len(weight_fp.shape) == 4:
            with torch.no_grad():
                range_w = 4 * torch.std(weight_fp, dim=(1, 2, 3)).detach()
            recons_w = (alpha * range_w / (2 + epsilon))[:, None, None, None] * w_q
        return recons_w
