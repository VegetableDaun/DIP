import tensorflow as tf

from STYLEGAN2.layers_generator.modulated_conv_2d_layer import ModulatedConv2DLayer


class ToRgbLayer(tf.keras.layers.Layer):
    """
    StyleGan2 generator To RGB layer
    """

    def __init__(self, **kwargs):
        super(ToRgbLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mod_conv2d_rgb = ModulatedConv2DLayer(filters=3,
                                                   kernel_size=1,
                                                   demod=False,
                                                   kernel_initializer=tf.keras.initializers.VarianceScaling(200 / input_shape[2]),
                                                   name='ToRGB')

        self.rgb_bias = self.add_weight(name='ToRGB/bias', shape=(3,),
                                        initializer=tf.random_normal_initializer(0, 1), trainable=True)

    def call(self, x, dlatent_vect, y):
        u = self.mod_conv2d_rgb([x, dlatent_vect])
        t = u + tf.reshape(self.rgb_bias, [-1 if i == (x.shape.rank - 1) else 1 for i in range(x.shape.rank)]) # ??????????????????

        return t if y is None else y + t
