from tensorflow.keras import layers
from tensorflow.keras import keras

# Create the CNN
input_img = layers.Input(shape=(224,224,3), name='ImageInput')

x = layers.Conv2D(32, (3,3), activation='selu', name='Conv2d_1')(input_img)
#x = layers_generator.BatchNormalization(name='bn_1')(x)
x = layers.MaxPooling2D((2,2), name='max_pool_1')(x)

x = layers.Conv2D(32, (3,3), activation='selu', name='Conv2d_2')(x)
#x = layers_generator.BatchNormalization(name='bn_2')(x)
x = layers.MaxPooling2D((2,2), name='max_pool_2')(x)

x = layers.Conv2D(64, (3,3), activation='selu', name='Conv2d_3')(x)
#x = layers_generator.BatchNormalization(name='bn_3')(x)
x = layers.MaxPooling2D((2,2), name='max_pool_3')(x)


x = layers.Flatten(name='flatten')(x)

x = layers.Dense(64, activation='selu', name='fc_1')(x)
x = layers.Dropout(0.7, name='dropout_1')(x)

x = layers.Dense(32, activation='selu', name='fc_2')(x)
x = layers.Dropout(0.3, name='dropout_2')(x)

x = layers.Dense(3, activation='softmax', name='fc_3')(x)

Classificator = keras.models.Model(inputs=input_img, outputs=x, name='CNN')

#kernel_regularizer='l2'