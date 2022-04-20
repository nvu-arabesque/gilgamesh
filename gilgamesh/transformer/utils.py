from enum import Enum
import numpy as np
import tensorflow as tf


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
