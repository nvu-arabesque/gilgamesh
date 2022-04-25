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
from gilgamesh.transformer.decoder_layer import DecoderLayer
from gilgamesh.transformer.positional_encoding import PositionalEncodingLayer


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
        self,
        decoder_layer: DecoderLayer,
        num_layers: int,
        d_embed: int,
        n_position: int = 200,
        norm=None,
    ):

        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for i in range(num_layers)]
        )
        self.position_enc = PositionalEncodingLayer(
            d_embed=d_embed, n_position=n_position
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args
        -------
        tgt: tensor
            the sequence to the decoder (required).
        memory: tensor
            the sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            the mask for the tgt sequence (optional).
        memory_mask: tensor
            the mask for the memory sequence (optional).
        tgt_key_padding_mask:
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask:
            the mask for the memory keys per batch (optional).
        """

        output = self.position_enc(tgt)
        for mod in self.layers:
            output, *_ = mod.forward(
                enc_output=memory,
                dec_input=output,
                slf_attn_mask=tgt_mask,
                dec_enc_attn_mask=memory_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
