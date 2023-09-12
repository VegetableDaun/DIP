from tensorflow.keras import layers
from tensorflow.keras import keras

# Create the discriminator.
input_img = layers.Input(shape=(224, 224, discriminator_in_channels), name='ImageInput')

x = layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_1')(
    input_img)
x = layers.BatchNormalization(name='Batch_1')(x)
# x = layers.Dropout(0.2)(x)

##x = layers.Conv2D(32, (1, 1), activation=layers.LeakyReLU(alpha=0.2), strides = (2, 2), padding="same", name='Conv2d_2')(input_img)
##x = layers.BatchNormalization(name='Batch_2')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_2')(
    x)
x = layers.BatchNormalization(name='Batch_2')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_3')(
    x)
x = layers.BatchNormalization(name='Batch_3')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_4')(
    x)
x = layers.BatchNormalization(name='Batch_4')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_5')(
    x)
x = layers.BatchNormalization(name='Batch_5')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(2, 2), padding="same", name='Conv2d_6')(
    x)
x = layers.BatchNormalization(name='Batch_6')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=(1, 1), padding="same", name='Conv2d_7')(
    x)
x = layers.BatchNormalization(name='Batch_7')(x)
# x = layers.Dropout(0.2)(x)

x = layers.Flatten(name='flatten')(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(1, activation='sigmoid', name='fc_1')(x)

Discriminator = keras.models.Model(inputs=input_img, outputs=x, name='Discriminator')