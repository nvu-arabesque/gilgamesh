"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
"""
import torch.nn as nn
from gilgamesh.transformer.feed_forward import PositionwiseFeedForward
from gilgamesh.transformer.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int = None,
        d_v: int = None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
        self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None
    ):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
