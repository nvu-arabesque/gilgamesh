"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
"""
from typing import Any, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gilgamesh.transformer.decoder import Decoder
from gilgamesh.transformer.decoder_layer import DecoderLayer

from gilgamesh.transformer.encoder import Encoder
from gilgamesh.transformer.encoder_layer import EncoderLayer
from gilgamesh.transformer.utils import (
    get_pad_mask,
    get_subsequent_mask,
    tuple_of_tensors_to_tensor,
)


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_inner: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        models_params = {
            "d_model": d_model,
            "n_head": n_head,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "d_inner": d_inner,
            "dropout": dropout,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
        }
        self.models_params = models_params
        self.encoder = Encoder(
            encoder_layer=EncoderLayer(**models_params),
            num_layers=models_params["num_encoder_layers"],
        )

        self.decoder = Decoder(
            decoder_layer=DecoderLayer(**models_params),
            num_layers=models_params["num_decoder_layers"],
        )
        self.ffnn = nn.Linear(d_model, 1, bias=False)

        self._reset_parameters()

        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, source_seq, target_seq):
        trg_mask = get_subsequent_mask(target_seq.shape[1])
        enc_output, _ = self.encoder(src=source_seq)
        dec_output = self.decoder.forward(
            tgt=target_seq,
            memory=enc_output,
            tgt_mask=trg_mask,
        )
        seq_logit = self.ffnn(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))
