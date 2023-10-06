import tensorflow as tf
import keras

from stylegan2_generator import StyleGan2Generator
from stylegan2_discriminator import StyleGan2Discriminator


class StyleGan2(tf.keras.Model):
    """ 
    StyleGan2 config f for tensorflow 2.x 
    """

    def __init__(self, resolution=1024, weights=None, latent_dim=512, **kwargs):
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
        super(StyleGan2, self).__init__(**kwargs)

        self.resolution = resolution
        if weights is not None:
            self.__adjust_resolution(weights)
        self.generator = StyleGan2Generator(resolution=self.resolution, weights=weights, name='Generator')
        self.discriminator = StyleGan2Discriminator(resolution=self.resolution, weights=weights, name='Discriminator')

        self.latent_dim = latent_dim
        # self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}

    def compile(self, d_optimizer, g_optimizer, loss_fn, *args, **kwargs):
        super(StyleGan2, self).compile(*args, **kwargs)

        # self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.loss_fn = loss_fn

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        return loss

    def wasserstein_loss(self, y_true, y_pred):
        return -tf.reduce_mean(y_true * y_pred)

    @tf.function
    def train_step(self, data):

        # Unpack the data.
        real_images, one_hot_labels = data

        batch_size = tf.shape(real_images)[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        # z is noise
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # generator
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(z)
            pred_fake = self.discriminator(fake_images)
            g_loss = self.wasserstein_loss(real_labels, pred_fake)

            gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # discriminator
        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:
            # forward pass
            pred_fake = self.discriminator(fake_images)
            pred_real = self.discriminator(real_images)

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon * real_images + (1 - epsilon) * fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator(interpolates)

            # calculate losses
            loss_fake = self.wasserstein_loss(fake_labels, pred_fake)
            loss_real = self.wasserstein_loss(real_labels, pred_real)
            loss_fake_grad = self.wasserstein_loss(fake_labels, pred_fake_grad)

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.loss_weights["gradient_penalty"] * self.gradient_loss(gradients_fake)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = self.loss_weights["drift"] * tf.reduce_mean(all_pred ** 2)

            d_loss = loss_fake + loss_real + gradient_penalty + drift_loss

            gradients = total_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def call(self, latent_vector):
        """
        Parameters
        ----------
        latent_vector : latent vector z of size [batch, 512].

        Returns
        -------
        score : output of the discriminator. 
        """
        img = self.generator(latent_vector)
        score = self.discriminator(img)

        return score

    def __adjust_resolution(self, weights_name):
        """
        Adjust resolution of the synthesis network output. 
        
        Parameters
        ----------
        weights_name : name of the weights
        """
        if weights_name == 'ffhq':
            self.resolution = 1024
        elif weights_name == 'car':
            self.resolution = 512
        elif weights_name in ['cat', 'church', 'horse']:
            self.resolution = 256
