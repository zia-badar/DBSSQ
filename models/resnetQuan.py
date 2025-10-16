import math
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import functional as F, Module, Parameter, init
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd, convolution_notes
from torch.nn.modules.utils import _pair
from torchvision.models import ResNet18_Weights
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.utils import _log_api_usage_once

__all__ = [
    "ResNetQuan",
    "resnet18Quan",
]

from models.batch_normalization import BatchNorm2d

class quantization(nn.Module):
    def __init__(self, q=4, s=0, type='n_bits', name='', grad_scaling=False):
        super().__init__()
        self.epsilon = 1
        self.q = q
        self.s = s
        self.grad_scaling = grad_scaling
        self.type = type
        self.name = name
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x):
        if self.epsilon == 1:
            return x

        if self.type == 'shift':
            return self.q_shift(x, None, self.epsilon, self.q, self.s)
        elif self.type == 'n_bits':
            return self.q_n(x, None, self.epsilon, self.q, self.s)

    def q_shift(self, x, std, epsilon, q, s):
        if q != 1:
            return None

        cond1 = x <= 0
        v = x * epsilon

        if s == 0:
            out = torch.where(cond1, v+(-1+epsilon), v+(1-epsilon))

        if s == 1:
            cond2 = torch.abs(x) <= 1/2**1

            out = torch.where(cond1,
                           torch.where(cond2, v+(1/2**1)*(-1+epsilon), v+(1/2**0)*(-1+epsilon)),
                           torch.where(cond2, v+(1/2**1)*(1-epsilon), v+(1/2**0)*(1-epsilon)))
        if s == 2:
            x = abs(x)
            cond2 = x <= 1/2**3
            cond3 = x <= 1/2**2
            cond4 = x <= 1/2**1

            v1 = torch.where(cond4, v+(1/2**1)*(-1+epsilon), v+(1/2**0)*(-1+epsilon))
            v2 = torch.where(cond3, v + (1 / 2 ** 2) * (-1 + epsilon), v1)
            v3 = torch.where(cond2, v + (1 / 2 ** 3) * (-1 + epsilon), v2)

            u1 = torch.where(cond4, v+(1/2**1)*(+1-epsilon), v+(1/2**0)*(+1-epsilon))
            u2 = torch.where(cond3, v+(1/2**2)*(+1-epsilon), u1)
            u3 = torch.where(cond2, v + (1 / 2 ** 3) * (+1 - epsilon), u2)

            out = torch.where(cond1, v3, u3)

        if s == 3:
            x = abs(x)
            cond2 = x <= 1/2**7
            cond3 = x <= 1/2**6
            cond4 = x <= 1/2**5
            cond5 = x <= 1/2**4
            cond6 = x <= 1/2**3
            cond7 = x <= 1/2**2
            cond8 = x <= 1/2**1

            v_ = torch.where(cond8, v+(1/2**1)*(-1+epsilon), v+(1/2**0)*(-1+epsilon))
            v_ = torch.where(cond7, v + (1 / 2 ** 2) * (-1 + epsilon), v_)
            v_ = torch.where(cond6, v + (1 / 2 ** 3) * (-1 + epsilon), v_)
            v_ = torch.where(cond5, v + (1 / 2 ** 4) * (-1 + epsilon), v_)
            v_ = torch.where(cond4, v + (1 / 2 ** 5) * (-1 + epsilon), v_)
            v_ = torch.where(cond3, v + (1 / 2 ** 6) * (-1 + epsilon), v_)
            v_ = torch.where(cond2, v + (1 / 2 ** 7) * (-1 + epsilon), v_)

            u_ = torch.where(cond8, v+(1/2**1)*(+1-epsilon), v+(1/2**0)*(+1-epsilon))
            u_ = torch.where(cond7, v + (1 / 2 ** 2) * (+1 - epsilon), u_)
            u_ = torch.where(cond6, v + (1 / 2 ** 3) * (+1 - epsilon), u_)
            u_ = torch.where(cond5, v + (1 / 2 ** 4) * (+1 - epsilon), u_)
            u_ = torch.where(cond4, v + (1 / 2 ** 5) * (+1 - epsilon), u_)
            u_ = torch.where(cond3, v + (1 / 2 ** 6) * (+1 - epsilon), u_)
            u_ = torch.where(cond2, v + (1 / 2 ** 7) * (+1 - epsilon), u_)

            out = torch.where(cond1, v_, u_)


        return out

    def q_n(self, x, std, epsilon, q, _none=None):
        _x = x * 2 ** (q - 1)
        cond1 = _x <= 0
        v = _x * epsilon

        if q == 1:
            out = torch.where(cond1, v+(-1+epsilon), v+(1-epsilon))

        if q == 2:
            cond2 = torch.abs(_x) <= 1

            out = torch.where(cond1,
                           torch.where(cond2, v+(1)*(-1+epsilon), v+(2)*(-1+epsilon)),
                           torch.where(cond2, v+(1)*(1-epsilon), v+(2)*(1-epsilon)))
        if q == 3:
            _x = abs(_x)
            cond2 = _x <= 1
            cond3 = _x <= 2
            cond4 = _x <= 3

            v_ = torch.where(cond4, v + (3) * (-1 + epsilon), v+(4)*(-1+epsilon))
            v_ = torch.where(cond3, v + (2) * (-1 + epsilon), v_)
            v_ = torch.where(cond2, v + (1) * (-1 + epsilon), v_)

            u_ = torch.where(cond4, v + (3) * (1 - epsilon), v+4*(1-epsilon))
            u_ = torch.where(cond3, v + (2) * (1 - epsilon), u_)
            u_ = torch.where(cond2, v + (1) * (1 - epsilon), u_)

            out = torch.where(cond1, v_, u_)

        if q == 4:
            _x = abs(_x)
            cond2 = _x <= 1
            cond3 = _x <= 2
            cond4 = _x <= 3
            cond5 = _x <= 4
            cond6 = _x <= 5
            cond7 = _x <= 6
            cond8 = _x <= 7

            v_ = torch.where(cond8, v+(7)*(-1+epsilon), v+(8)*(-1+epsilon))
            v_ = torch.where(cond7, v + (6) * (-1 + epsilon), v_)
            v_ = torch.where(cond6, v + (5) * (-1 + epsilon), v_)
            v_ = torch.where(cond5, v + (4) * (-1 + epsilon), v_)
            v_ = torch.where(cond4, v + (3) * (-1 + epsilon), v_)
            v_ = torch.where(cond3, v + (2) * (-1 + epsilon), v_)
            v_ = torch.where(cond2, v + (1) * (-1 + epsilon), v_)

            u_ = torch.where(cond8, v+(7)*(+1-epsilon), v+(8)*(+1-epsilon))
            u_ = torch.where(cond7, v + (6) * (+1 - epsilon), u_)
            u_ = torch.where(cond6, v + (5) * (+1 - epsilon), u_)
            u_ = torch.where(cond5, v + (4) * (+1 - epsilon), u_)
            u_ = torch.where(cond4, v + (3) * (+1 - epsilon), u_)
            u_ = torch.where(cond3, v + (2) * (+1 - epsilon), u_)
            u_ = torch.where(cond2, v + (1) * (+1 - epsilon), u_)

            out = torch.where(cond1, v_, u_)

        out /= 2 ** (q - 1)
        return out

    def _backward_hook(self, module, grad_input, grad_output):
        if self.grad_scaling:
            return (grad_input[0] / module.epsilon,)
        else:
            return grad_input


class LinearQuan(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class Conv2dQuan(_ConvNd):
    __doc__ = (
        r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.
"""
        """
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the \u00e0 trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
"""
        r"""

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """.format(
            **reproducibility_notes, **convolution_notes
        )
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        use_weight_quantization: bool,
        use_activation_quantization: bool,
        activation_already_normalized: bool = False,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.activation_already_normalized = activation_already_normalized
        self.fixed_std = torch.ones(out_channels).cuda()
        self.use_weight_quantization = use_weight_quantization
        self.use_activation_quantization = use_activation_quantization

        if self.use_weight_quantization:
            self.weight_quantization_module = quantization(name='weight')
        if self.use_activation_quantization:
            self.activation_quantization_module = quantization(name='activation')

        self.save_activations = False
        self.last_fp_activation = None
        self.last_quan_activation = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        if not self.activation_already_normalized:
            if not hasattr(self, 'activation_net'):
                self.activation_net = BatchNorm2d(input.shape[1]).to(input.device)
                print(f"initialized activation_network")

            input = self.activation_net(input)

        if self.use_activation_quantization:
            input = self.activation_quantization_module(input / (3))
            input = input * 3

        if self.use_weight_quantization:
            with torch.no_grad():
                mean = weight.mean(dim=(1, 2, 3))[:, None, None, None]
                std = torch.std(weight, dim=(1, 2, 3))[:, None, None, None]
            weight_norm = (weight - mean) / std

            weight_q = self.weight_quantization_module(weight_norm / 3) * (3 * self.fixed_std[:, None, None, None])

            return F.conv2d(
                input, weight_q, bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            return F.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )


    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

def conv3x3(in_planes: int, out_planes: int, use_weight_quantization: bool, use_activation_quantization: bool, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2dQuan:
    """3x3 convolution with padding"""
    return Conv2dQuan(
        in_planes,
        out_planes,
        kernel_size=3,
        use_weight_quantization=use_weight_quantization,
        use_activation_quantization=use_activation_quantization,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, use_weight_quantization: bool, use_activation_quantization: bool, stride: int = 1) -> Conv2dQuan:
    """1x1 convolution"""
    return Conv2dQuan(in_planes, out_planes, kernel_size=1, use_weight_quantization=False, use_activation_quantization=False, activation_already_normalized=True, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        use_weight_quantization: bool,
        use_activation_quantization: bool,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, use_weight_quantization, use_activation_quantization, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, use_weight_quantization, use_activation_quantization)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetQuan(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_weight_quantization=True,
        use_activation_quantization=True,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.use_weight_quantization = use_weight_quantization
        self.use_activation_quantization = use_activation_quantization
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2dQuan(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, use_weight_quantization=False, use_activation_quantization=False, activation_already_normalized=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearQuan(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        with torch.no_grad():
            self.eval()  # this is required so that next inference do not update the bn layers.
            self.cuda()
            self(torch.ones((3, 3, 224, 224)).to(torch.float).cuda())

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, False, False, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, self.use_weight_quantization, self.use_activation_quantization, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    self.use_weight_quantization,
                    self.use_activation_quantization,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnetQuan(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNetQuan:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetQuan(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True), strict=False)

    return model

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18Quan(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNetQuan:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnetQuan(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)