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


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int = None,
        d_v: int = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        d_k_default = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k if d_k else d_k_default
        self.d_v = d_v if d_v else d_k_default
        assert (
            self.d_k * self.n_head == self.d_model
        ), f"d_k: {d_k} * n_head {n_head} != d_model: {d_model}"
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
