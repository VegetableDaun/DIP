import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from config_GAN import latent_dim, path_to_result, batch_size
from scipy import linalg


class FID:
    def __init__(self, generator=None, data=None, step_gen=None):
        self.gen_mu = None
        self.gen_sigma = None
        self.generator = generator

        self.inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                                 weights="imagenet",
                                                                 pooling='avg')

        self.path_to_result = Path(path_to_result) / 'real_mu_sigma.json'
        self.data = data
        self.step_gen = step_gen

        if os.path.isfile(self.path_to_result):
            with open(self.path_to_result, 'r') as F:
                real_mu_sigma = list(json.load(F))
            self.real_mu = np.array(real_mu_sigma[0])
            self.real_sigma = np.array(real_mu_sigma[1])

        elif self.data is not None:
            real_embeddings = np.zeros([1, 2048])
            for i in self.data:
                # we need to have values between 1 and 255
                i = i[0] * 255

                # [optional]: we may need 3 channel (instead of 1)
                # i = np.repeat(i, 3, axis=-1)

                # resize the input shape , i.e. old shape: 32, new shape: 112
                i = tf.image.resize(i, [128, 128])  # if we want to resize

                # round values
                i = tf.round(i)

                # we need to have values between -1 and 1
                i = tf.keras.applications.inception_v3.preprocess_input(i)
                predicted_img = self.inception_model.predict(i, verbose=0)
                real_embeddings = np.vstack((real_embeddings, predicted_img))

            self.real_mu, self.real_sigma = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
            with open(self.path_to_result, 'w') as F:
                json.dump([self.real_mu.tolist(), self.real_sigma.tolist()], F)
        else:
            raise Exception

    def calculate_fid(self, path_to_generator):
        # generate image
        self.generator.load(path_to_generator)

        generated_embeddings = np.zeros([1, 2048])
        for i in self.data:
            # Create noise vector with label
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            vector_labels = i[1]

            # Generating the images
            generated_img = self.generator(latent_vectors, c=vector_labels)

            # Preparing generated_img for inceptionV3
            # We need to have values between 1 and 255
            generated_img = generated_img * 255

            # [optional]: we may need 3 channel (instead of 1)
            # generated_img = np.repeat(generated_img, 3, axis=-1)

            # resize the input shape , i.e. old shape: 32, new shape: 112
            generated_img = tf.image.resize(generated_img, [128, 128])  # if we want to resize

            # round values
            generated_img = tf.round(generated_img)

            # we need to have values between -1 and 1
            generated_img = tf.keras.applications.inception_v3.preprocess_input(generated_img)

            # inceptionV3 predicts
            predicted_img = self.inception_model.predict(generated_img, verbose=1)
            generated_embeddings = np.vstack((generated_embeddings, predicted_img))

            if self.step_gen is None:
                continue
            elif generated_embeddings.shape[0] > self.step_gen:
                break

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
