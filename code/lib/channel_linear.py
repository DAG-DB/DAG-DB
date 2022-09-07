import math

import torch
from torch import nn


class ChannelLinear(nn.Module):
    """ The difference from nn.Linear is that each channel defines, in effect,
    a separate linear layer

    In dag_learner.fNet, the channel corresponds to the child node,
    i.e. each child has a separate linear net (with input that will be from
    parents)

    For comparison with nn.Linear, see
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    def __init__(self, n_channels, bias_flag, n_in, n_out):
        super().__init__()
        self.n_channels = n_channels
        self.n_in = n_in
        self.n_out = n_out
        self.weight = nn.Parameter(torch.empty((n_out, n_in, n_channels)))
        self.bias_flag = bias_flag
        if bias_flag:
            self.bias = nn.Parameter(torch.empty(n_out, n_channels))
        else:
            self.bias = None
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        if self.bias_flag:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = torch.einsum(
            'bsic, oic -> bsoc', input, self.weight)
        if self.bias_flag:
            output += self.bias[None, None, :, :]
        return output

    def __repr__(self) -> str:
        return f'ChannelLinear(n_channels={self.n_channels}, ' \
               f'n_in={self.n_in}, n_out={self.n_out}, ' \
               f'bias=self.bias_flag)'


if __name__ == '__main__':
    cl = ChannelLinear(3, True, 5, 7)
    pass
