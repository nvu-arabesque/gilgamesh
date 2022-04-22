"""
@credit: 
- mchen@arabesque.ai
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- https://arxiv.org/pdf/1706.03762.pdf
"""

from enum import Enum
import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable

from gilgamesh.transformer.data import Batch


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(size):
    """ Returns lower triangle, the upper fille with -inf for masking attention strictly up to time step t. """
    return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)


def positional_encoding(position, d_model):
    """Returns the positional encoding for transformer

    Original author: Mabelle Chen mchen@arabesque.a
    """

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, d_ff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                d_ff, activation=tf.nn.relu
            ),  # (batch_size, seq_len, d_ff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


def data_gen(
    batch_size: int = 10,
    seq_length: int = 32,
    embed_dim: int = 512,
    n_batches: int = 10,
    V: int = 100,
):
    "Generate random data for a src-tgt copy task."
    for i in range(n_batches):
        data = torch.from_numpy(
            np.random.randint(1, V, size=(batch_size, seq_length, embed_dim))
        )
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt)
