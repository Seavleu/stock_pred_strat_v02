import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        pos_encoding = tf.cast(self.pos_encoding, dtype=inputs.dtype)
        return inputs + pos_encoding[:, :tf.shape(inputs)[1], :]

    def compute_output_shape(self, input_shape):
        return input_shape
