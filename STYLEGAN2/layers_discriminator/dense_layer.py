import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    """
    StyleGan2 Dense layer, including weights multiplication per runtime coef, and bias multiplication per lrmul
    """

    def __init__(self, fmaps, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.fmaps = fmaps

    def build(self, input_shape):
        self.dense_weight = self.add_weight(name='weight', shape=(input_shape[1], self.fmaps),
                                            initializer=tf.random_normal_initializer(0, 200 / input_shape[2]), trainable=True)
        self.dense_bias = self.add_weight(name='bias', shape=(self.fmaps,),
                                          initializer=tf.random_normal_initializer(0, 200 / input_shape[2]), trainable=True)

    def call(self, x):
        x = tf.matmul(x, tf.math.multiply(self.dense_weight, self.runtime_coef))
        x += tf.reshape(tf.math.multiply(self.dense_bias, self.lrmul),
                        [-1 if i == 1 else 1 for i in range(x.shape.rank)])

        return x
