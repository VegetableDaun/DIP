from keras import layers
import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    """
    StyleGan2 Dense layer, including weights multiplication per runtime coef, and bias multiplication per lrmul
    """

    def __init__(self, fmaps, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.Dense = None
        self.LeakyRelu = None
        self.fmaps = fmaps

    def build(self, input_shape):
        self.Dense = layers.Dense(self.fmaps, input_shape=[input_shape[1]])
        self.LeakyRelu = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.Dense(x)
        x = self.LeakyRelu(x)

        return x
