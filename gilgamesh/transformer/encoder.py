"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
"""
import copy
from typing import Optional
from torch import Tensor
import torch.nn as nn

from gilgamesh.transformer.encoder_layer import EncoderLayer
from gilgamesh.transformer.feed_forward import PositionwiseFeedForward
from gilgamesh.transformer.multi_head_attention import MultiHeadAttention


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, encoder_layer: EncoderLayer, num_layers: int, norm=None):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        attentions = []
        for mod in self.layers:
            output, attn = mod.forward(output, slf_attn_mask=mask)
            attentions.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attentions
