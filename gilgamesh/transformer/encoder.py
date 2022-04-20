"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
- https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
"""
import torch.nn as nn
from gilgamesh.transformer.encoder_layer import EncoderLayer
from gilgamesh.transformer.feed_forward import PositionwiseFeedForward
from gilgamesh.transformer.multi_head_attention import MultiHeadAttention
from gilgamesh.transformer.utils import positional_encoding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
        self,
        n_src_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=0.1,
        n_position=200,
    ):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = positional_encoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

def _enc(self, src_seq):
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)
        return enc_output

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        enc_output = self._enc(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return (enc_output,)
