"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
"""
from typing import Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(
        self,
        d_in,
        d_hid,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        dropout=0.1,
        layer_norm_eps=1e-6,
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=layer_norm_eps)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        # original paper does not use dropout1 but torch does
        x = self.dropout2(self.w_2(self.activation(self.w_1(x))))
        x += residual
        x = self.layer_norm(x)
        return x
