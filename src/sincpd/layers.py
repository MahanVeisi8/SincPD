import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class SincConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, sample_rate, min_low_hz=0.0, min_band_hz=1.0, **kwargs):
        super().__init__(**kwargs)
        assert kernel_size % 2 == 1
        self.filters = filters
        self.kernel_size = kernel_size
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)
        self._t = tf.constant(np.arange(-(kernel_size//2), kernel_size//2+1, dtype=np.float32))
    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.low_hz = tf.Variable(tf.random.uniform([self.filters], minval=self.min_low_hz, maxval=self.sample_rate/4.0), trainable=True, dtype=tf.float32)
        self.band_hz = tf.Variable(tf.random.uniform([self.filters], minval=self.min_band_hz, maxval=self.sample_rate/4.0), trainable=True, dtype=tf.float32)
        self.channel_weights = self.add_weight(name='channel_weights', shape=(in_channels, self.filters), initializer='glorot_uniform', trainable=True)
    def _sinc(self, x):
        eps = 1e-7
        return tf.sin(np.pi * (x + eps)) / (np.pi * (x + eps))
    def call(self, inputs):
        low = tf.abs(self.low_hz) + self.min_low_hz
        high = tf.clip_by_value(low + tf.abs(self.band_hz) + self.min_band_hz, 0.0, self.sample_rate/2.0)
        low_norm, high_norm = low / self.sample_rate, high / self.sample_rate
        t = tf.reshape(tf.cast(self._t, tf.float32), [1, -1])
        f1, f2 = tf.reshape(low_norm, [-1,1]), tf.reshape(high_norm, [-1,1])
        h1 = 2.0 * f2 * self._sinc(2.0 * f2 * t)
        h2 = 2.0 * f1 * self._sinc(2.0 * f1 * t)
        h = h1 - h2
        K = tf.shape(h)[1]
        n = tf.cast(tf.range(K), tf.float32)
        w = 0.54 - 0.46 * tf.cos(2.0 * np.pi * n / tf.cast(K-1, tf.float32))
        h = (h * w) / (tf.reduce_sum(tf.abs(h), axis=1, keepdims=True) + 1e-6)
        inC = inputs.shape[-1]
        h = tf.transpose(h, [1,0])
        kernel = tf.tile(tf.expand_dims(h,1), [1,inC,1]) * tf.expand_dims(self.channel_weights,0)
        return tf.nn.conv1d(inputs, filters=kernel, stride=1, padding='SAME')