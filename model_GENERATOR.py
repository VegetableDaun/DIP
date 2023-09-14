from tensorflow import keras
from keras import layers

from config_GAN import generator_in_channels

# Create the generator.
input_img = layers.Input(shape=(generator_in_channels), name='NoiseInput')

x = layers.Dense(14 * 14 * generator_in_channels, activation=layers.LeakyReLU(alpha=0.2), name='fc_1')(input_img)
x = layers.Reshape((14, 14, generator_in_channels))(x)
x = layers.BatchNormalization(name='Batch_1')(x)

x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same", activation=layers.LeakyReLU(alpha=0.2),
                           name='Conv2DTrans_1')(x)
x = layers.BatchNormalization(name='Batch_2')(x)

x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation=layers.LeakyReLU(alpha=0.2),
                           name='Conv2DTrans_2')(x)
x = layers.BatchNormalization(name='Batch_3')(x)

x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", activation=layers.LeakyReLU(alpha=0.2),
                           name='Conv2DTrans_3')(x)
x = layers.BatchNormalization(name='Batch_4')(x)

x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", activation=layers.LeakyReLU(alpha=0.2),
                           name='Conv2DTrans_4')(x)
x = layers.BatchNormalization(name='Batch_5')(x)

x = layers.Conv2D(3, (5, 5), padding="same", activation="sigmoid", name='Conv2D_1')(x)

Generator = keras.models.Model(inputs=input_img, outputs=x, name='Generator')
