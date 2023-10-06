import keras
import tensorflow as tf
import numpy as np

from STYLEGAN2.layers_generator.dense_layer import DenseLayer
from STYLEGAN2.layers_generator.synthesis_main_layer import SynthesisMainLayer
from STYLEGAN2.layers_generator.to_rgb_layer import ToRgbLayer
from STYLEGAN2.utils_stylegan2 import nf
from STYLEGAN2.weights_map import available_weights, synthesis_weights, mapping_weights, weights_stylegan2_dir


class MappingNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator mapping network, from z to dlatents for tensorflow 2.x
    """

    def __init__(self, resolution=1024, **kwargs):
        super(MappingNetwork, self).__init__(**kwargs)
        self.dlatent_size = 512
        self.mapping_layers = 8

        self.dlatent_vector = (int(np.log2(resolution))-1)*2

    def build(self, input_shape):
        self.S = keras.models.Sequential()
        for i in range(self.mapping_layers):
            self.S.add(DenseLayer(fmaps=512, name='Dense{}'.format(i)))

        self.g_mapping_broadcast = tf.keras.layers.RepeatVector(self.dlatent_vector)

    def call(self, z):
        x = tf.cast(z, 'float32')
        x = self.S(x)

        # Broadcasting
        dlatents = self.g_mapping_broadcast(x)

        return dlatents


class SynthesisNetwork(tf.keras.layers.Layer):
    """
    StyleGan2 generator synthesis network from dlatents to img tensor for tensorflow 2.x
    """

    def __init__(self, resolution=1024, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed to the floor integer power of 2.
            The default is 1024.
        """
        super(SynthesisNetwork, self).__init__(**kwargs)
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(self.resolution))

    def build(self, input_shape):

        # constant layer
        self.const_4_4 = self.add_weight(name='4x4/Const/const', shape=(1, 4, 4, 512),
                                         initializer=tf.random_normal_initializer(0, 1), trainable=True)
        # early layer 4x4
        self.Dense_rgb_style_4_4 = keras.layers.Dense(nf(1, fmap_decay=100), name='rgb_style_4x4')

        self.layer_4_4 = SynthesisMainLayer(filter=nf(1, fmap_decay=100), name='4x4')
        self.torgb_4_4 = ToRgbLayer(name='4x4')

        # main layers_generator
        for res in range(3, self.resolution_log2 + 1):
            res_str = str(2 ** res)
            setattr(self, 'Dense_rgb_style_{}_{}'.format(res_str, res_str),
                    keras.layers.Dense(nf(res - 1, fmap_decay=100), name='rgb_style_{}{}'.format(res_str, res_str)))
            setattr(self, 'Dense_style_1_{}_{}'.format(res_str, res_str),
                    keras.layers.Dense(nf(res - 1, fmap_decay=100), name='style_{}x{}'.format(res_str, res_str)))
            setattr(self, 'Dense_style_2_{}_{}'.format(res_str, res_str),
                    keras.layers.Dense(nf(res - 1, fmap_decay=100), name='style_{}x{}'.format(res_str, res_str)))

            setattr(self, 'layer_{}_{}_up'.format(res_str, res_str),
                    SynthesisMainLayer(filter=nf(res - 1, fmap_decay=100), up=True, name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'layer_{}_{}'.format(res_str, res_str),
                    SynthesisMainLayer(filter=nf(res - 1, fmap_decay=100), name='{}x{}'.format(res_str, res_str)))
            setattr(self, 'torgb_{}_{}'.format(res_str, res_str),
                    ToRgbLayer(name='{}x{}'.format(res_str, res_str)))

    def call(self, dlatents_in):

        dlatents_in = tf.cast(dlatents_in, 'float32')
        y = None

        # Early layers_generator
        x = tf.tile(tf.cast(self.const_4_4, 'float32'), [tf.shape(dlatents_in)[0], 1, 1, 1])
        x = self.layer_4_4(x, dlatents_in[:, 0])

        rgb_style = self.Dense_rgb_style_4_4(dlatents_in[:, 1])
        y = self.torgb_4_4(x, rgb_style, y)

        # Main layers_generator
        for res in range(3, self.resolution_log2 + 1):

            style_1 = getattr(self, 'Dense_style_1_{}_{}'.format(2 ** res, 2 ** res))(dlatents_in[:, res * 2 - 5])
            style_2 = getattr(self, 'Dense_style_2_{}_{}'.format(2 ** res, 2 ** res))(dlatents_in[:, res * 2 - 4])
            rgb_style = getattr(self, 'Dense_rgb_style_{}_{}'.format(2 ** res, 2 ** res))(dlatents_in[:, res * 2 - 3])


            x = getattr(self, 'layer_{}_{}_up'.format(2 ** res, 2 ** res))(x, style_1)
            x = getattr(self, 'layer_{}_{}'.format(2 ** res, 2 ** res))(x, style_2)

            y = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(y)
            y = getattr(self, 'torgb_{}_{}'.format(2 ** res, 2 ** res))(x, rgb_style, y)

        images_out = y
        return tf.identity(images_out, name='images_out')


class StyleGan2Generator(tf.keras.layers.Layer):
    """
    StyleGan2 generator config f for tensorflow 2.x
    """

    def __init__(self, resolution=1024, weights=None, **kwargs):
        """
        Parameters
        ----------
        resolution : int, optional
            Resolution output of the synthesis network, will be parsed
            to the floor integer power of 2.
            The default is 1024.
        weights : string, optional
            weights name in weights dir to be loaded. The default is None.
        """
        super(StyleGan2Generator, self).__init__(**kwargs)

        self.resolution = resolution
        if weights is not None: self.__adjust_resolution(weights)

        self.mapping_network = MappingNetwork(resolution=self.resolution, name='Mapping_network')
        self.synthesis_network = SynthesisNetwork(resolution=self.resolution, name='Synthesis_network')

        # load weights
        if weights is not None:
            # we run the network to define it, not the most efficient thing to do...
            _ = self(tf.zeros(shape=(1, 512)))
            self.__load_weights(weights)

    def call(self, z):
        """
        Parameters
        ----------
        z : tensor, latent vector of shape [batch, 512]

        Returns
        -------
        img : tensor, image generated by the generator of shape  [batch, channel, height, width]
        """
        dlatents = self.mapping_network(z)
        img = self.synthesis_network(dlatents)

        return img

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output.

        Parameters
        ----------
        weights_name : name of the weights

        Returns
        -------
        None.

        """
        if weights_name == 'ffhq':
            self.resolution = 1024
        elif weights_name == 'car':
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']:
            self.resolution = 256

    def __load_weights(self, weights_name):
        """
        Load pretrained weights, stored as a dict with numpy arrays.
        Parameters
        ----------
        weights_name : name of the weights

        Returns
        -------
        None.

        """

        if (weights_name in available_weights) and type(weights_name) == str:
            data = np.load(weights_stylegan2_dir + weights_name + '.npy', allow_pickle=True)[()]

            weights_mapping = [data.get(key) for key in mapping_weights]
            weights_synthesis = [data.get(key) for key in synthesis_weights[weights_name]]

            self.mapping_network.set_weights(weights_mapping)
            self.synthesis_network.set_weights(weights_synthesis)

            print("Loaded {} generator weights!".format(weights_name))
        else:
            raise Exception('Cannot load {} weights'.format(weights_name))
