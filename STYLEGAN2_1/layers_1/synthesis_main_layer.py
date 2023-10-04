import tensorflow as tf

from STYLEGAN2_1.layers_1.modulated_conv_2d_layer import ModulatedConv2DLayer


class SynthesisMainLayer(tf.keras.layers.Layer):
    """
    StyleGan2 synthesis network main layer
    """

    def __init__(self, filter, up=False, **kwargs):
        super(SynthesisMainLayer, self).__init__(**kwargs)

        self.filter = filter
        self.up = up
        self.kernel = 3

        self.resample_kernel = [1, 3, 3, 1]
        if self.up:
            self.l_name = 'Conv0_up'
        else:
            self.l_name = 'Conv1'

    def build(self, input_shape):

        self.noise_strength = self.add_weight(name=self.l_name + '/noise_strength', shape=[],
                                              initializer=tf.initializers.zeros(), trainable=True)
        self.bias = self.add_weight(name=self.l_name + '/bias', shape=(self.filter,),
                                    initializer=tf.random_normal_initializer(0, 1), trainable=True)

        self.mod_conv2d_layer = ModulatedConv2DLayer(kernel_size=self.kernel,
                                                     filters=self.filter,
                                                     name=self.l_name,
                                                     padding='same',
                                                     kernel_initializer='he_uniform')

    def call(self, x, dlatent_vect):
        # Upsampling
        if self.up:
            x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)

        x = self.mod_conv2d_layer([x, dlatent_vect])

        # randomize noise
        noise = tf.random.normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)

        # adding noise to layer
        x += tf.math.multiply(noise, tf.cast(self.noise_strength, x.dtype))

        # adding bias and lrelu activation
        x += tf.reshape(self.bias, [-1 if i == 1 else 1 for i in range(x.shape.rank)])
        x = tf.math.multiply(tf.nn.leaky_relu(x, 0.2), tf.math.sqrt(2.))  # ?????????????

        return x
