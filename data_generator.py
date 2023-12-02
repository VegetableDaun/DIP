import numpy as np
import tensorflow as tf
from tensorflow import keras
from helped_functions import increase, conv_RGB

from config_GAN import latent_dim, image_size, num_classes

def data_gen(data, batch_size, aug=None, add_labels=None, gen=None, gen_labels=None, latent_dim=latent_dim):
    if aug is not None and add_labels != {0: 0, 1: 0, 2: 0}:
        data = increase(data, add_labels)

    data = data.sample(frac=1.).reset_index(drop=True)  # Shuffle data

    n = len(data)  # Get total number of samples in the data
    indices = np.arange(n)  # Get a numpy array of all the indices of the input data
    steps = n // batch_size  # Get numbers of steps

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, num_classes), dtype=np.float32)

    i = 0  # Initialize a counter
    np.random.shuffle(indices)  # Random indices
    while True:
        # Get the next batch
        count = 0
        next_batch = indices[i * batch_size: (i + 1) * batch_size]
        for idx in next_batch:
            sample = data.iloc[int(idx)]
            if sample['add'] == 0:
                img, label = conv_RGB(sample['image'], sample['label'])  # generating samples

                batch_data[count] = img.astype(np.float32) / 255.
                batch_labels[count] = label

                count += 1
            else:
                img, label = conv_RGB(sample['image'], sample['label'])  # generating more samples

                batch_data[count] = aug(image=img)['image'].astype(np.float32) / 255.
                batch_labels[count] = label

                count += 1

        if gen is not None:
            if gen_labels is None:
                interpolation_noise = tf.random.normal(shape=(batch_data.shape[0], latent_dim))
                gen_batch_labels = batch_labels
            else:
                interpolation_noise = tf.random.normal(shape=(sum(gen_labels.values()), latent_dim))
                gen_batch_labels = np.repeat(list(gen_labels.keys()), list(gen_labels.values()))
                gen_batch_labels = keras.utils.to_categorical(gen_batch_labels, num_classes=num_classes)

            interpolation_noise_labels = tf.concat([interpolation_noise, gen_batch_labels], axis=1)
            gen_images = gen.predict(interpolation_noise_labels)

            batch_data = np.vstack((batch_data, gen_images))
            batch_labels = np.vstack((batch_labels, gen_batch_labels))

        yield batch_data, batch_labels

        i += 1
        if i == steps:
            i = 0
            np.random.shuffle(indices)
