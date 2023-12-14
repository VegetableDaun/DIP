import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from config_GAN import latent_dim, path_to_result, batch_size
from scipy import linalg


class FID:
    def __init__(self, generator=None, data=None, step_gen=None, len=None):
        self.gen_mu = None
        self.gen_sigma = None
        self.generator = generator
        self.len = len

        self.inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                                 weights="imagenet",
                                                                 pooling='avg')

        self.path_to_result = Path(path_to_result) / 'real_mu_sigma.json'
        self.data = data
        self.step_gen = step_gen

        if os.path.isfile(self.path_to_result):
            data = np.load(self.path_to_result, allow_pickle=True)[()]

            self.real_mu = data['real_mu']
            self.real_sigma = data['real_sigma']

        elif self.data is not None:
            j = 0

            real_embeddings = np.zeros([1, 2048])
            for i in self.data:
                # we need to have values between 1 and 255
                i = i[0] * 255

                # [optional]: we may need 3 channel (instead of 1)
                if tf.shape(i)[2] == 1:
                    i = tf.repeat(i, 3, axis=-1)

                # resize the input shape , i.e. old shape: 32, new shape: 256
                i = tf.image.resize(i, [256, 256])  # if we want to resize

                # round values
                i = tf.round(i)

                # we need to have values between -1 and 1
                i = tf.keras.applications.inception_v3.preprocess_input(i)
                predicted_img = self.inception_model.predict(i, verbose=0)
                real_embeddings = np.vstack((real_embeddings, predicted_img))

                if j >= self.len:
                    break
                j += tf.shape(i)[0]

            self.real_mu, self.real_sigma = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)

            data = {'real_mu': self.real_mu, 'real_sigma': self.real_sigma}
            np.save(self.path_to_result, data, allow_pickle=True)

        else:
            raise Exception

    def calculate_fid(self, path_to_generator):
        # generate image
        self.generator.load(path_to_generator)

        generated_embeddings = np.zeros([1, 2048])
        for i in self.data:
            n = 1
            j = 0

            if len(self.data) < self.step_gen:
                n = math.ceil(len(self.data) / self.step_gen + 1)

            # Create noise vector with label
            latent_vectors = tf.random.normal(shape=(n * batch_size, latent_dim))
            vector_labels = tf.repeat(i[1], n, axis=0)

            # Generating the images
            generated_img = self.generator(latent_vectors, c=vector_labels)
            generated_img = tf.transpose(generated_img, [0, 2, 3, 1])

            # Preparing generated_img for inceptionV3
            # We need to have values between 1 and 255
            generated_img = generated_img * 255

            # [optional]: we may need 3 channel (instead of 1)
            # generated_img = np.repeat(generated_img, 3, axis=-1)

            # resize the input shape , i.e. old shape: 256, new shape: 256
            generated_img = tf.image.resize(generated_img, [256, 256])  # if we want to resize

            # round values
            generated_img = tf.round(generated_img)

            # we need to have values between -1 and 1
            generated_img = tf.keras.applications.inception_v3.preprocess_input(generated_img)

            # inceptionV3 predicts
            predicted_img = self.inception_model.predict(generated_img, verbose=0)
            generated_embeddings = np.vstack((generated_embeddings, predicted_img))

            # if self.step_gen is None:
            #     continue
            # elif generated_embeddings.shape[0] > self.step_gen:
            #     break

            if j >= self.len:
                break
            j += tf.shape(i)[0] / n

        # calculate mean and covariance statistics
        self.gen_mu, self.gen_sigma = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)

        # calculate sum squared difference between means
        ssdiff = np.sum((self.real_mu - self.gen_mu) ** 2.0)

        # calculate sqrt of product between cov
        cov_mean = linalg.sqrtm(self.real_sigma.dot(self.gen_sigma))

        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        # calculate score
        fid = ssdiff + np.trace(self.real_sigma + self.gen_sigma - 2.0 * cov_mean)
        return fid
