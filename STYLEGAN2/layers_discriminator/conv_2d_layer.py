import keras.layers
import tensorflow as tf


class Conv2DLayer(tf.keras.layers.Layer):
    """
    StyleGan2 discriminator convolutional layer
    """

    def __init__(self, fmaps, kernel, up=False, down=False,
                 demodulate=True, gain=1, use_wscale=True, lrmul=1, **kwargs):

        super(Conv2DLayer, self).__init__(**kwargs)

        self.fmaps = fmaps
        self.kernel = kernel

        self.up = up
        self.down = down
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=(self.kernel, self.kernel, input_shape[1], self.fmaps),
                                      initializer=tf.random_normal_initializer(0, 200 / input_shape[1]), trainable=True)

    def call(self, x):
        # Convolution with optional up/downsampling.
        if self.up:
            x = tf.transpose(x, [0, 2, 3, 1])  # 'NCHW -> NHWC'
            x = keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', interpolation="bilinear")(x)
            x = tf.transpose(x, [0, 3, 1, 2])
        elif self.down:
            x = tf.transpose(x, [0, 2, 3, 1])  # 'NCHW -> NHWC'
            x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
            x = tf.transpose(x, [0, 3, 1, 2])
        else:
            x = tf.transpose(x, [0, 2, 3, 1])  # 'NCHW -> NHWC'
            x = keras.layers.Conv2D(filters=self.fmaps, kernel_size=self.kernel, kernel_initializer='he_uniform')(x)
            x = tf.transpose(x, [0, 3, 1, 2])

        return x
