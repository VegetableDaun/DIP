import numpy as np
import os
import json

from tensorflow import keras
from PIL import Image
from pathlib import Path
from config_GAN import path_to_result, path_to_noise, path_to_discriminator, path_to_generator


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, num_save=5, save_last=True, path=Path(path_to_result), noise=Path(path_to_noise)):
        super(CustomCallback, self).__init__()
        self.num_save = num_save
        self.path = path
        self.save_last = save_last

        try:
            os.makedirs(self.path)
        except:
            pass

        if not os.path.isfile(self.path / 'data.json'):
            self.counter = 0
        else:
            with open(self.path / 'data.json', 'r') as F:
                self.counter = int(json.load(F))

        with open(noise, mode='r') as F:
            self.noise = np.array(json.load(F))

    def on_train_begin(self, logs=None):
        if os.path.isfile(self.path / 'metrics.json'):
            self.counter -= self.counter % self.num_save

        if self.counter == 0:
            os.remove(self.path / 'metrics.json')

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        if not os.path.isfile(self.path / 'metrics.json'):
            with open(self.path / 'metrics.json', mode='w') as F:
                json.dump({self.counter: logs}, F)
        else:
            with open(self.path / 'metrics.json', mode='r') as F:
                feeds = dict(json.load(F))
                feeds[str(self.counter)] = logs
            with open(self.path / 'metrics.json', mode='w') as F:
                json.dump(feeds, F)

        if (epoch + 1) % self.num_save == 0:
            try:
                os.makedirs(self.path / 'Image')
                os.makedirs(self.path / path_to_discriminator)
                os.makedirs(self.path / path_to_generator)
            except:
                pass

            if self.save_last:
                self.model.discriminator.save(self.path / 'Discriminator.hdf5')
                self.model.generator.save(self.path / 'Generator.hdf5')
            else:
                self.model.discriminator.save(self.path / path_to_discriminator / f'Discriminator_{self.counter}.hdf5')
                self.model.generator.save(self.path / path_to_generator / f'Generator_{self.counter}.hdf5')

            img = np.array(self.model.generator(self.noise))[0]
            img = np.round(img * 255)
            img = img.astype(np.uint8)
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])
            img = Image.fromarray(img)
            img.save(self.path / ('Image/Epoch_' + f'{self.counter}.jpg'))

        with open(self.path / 'data.json', 'w') as F:
            json.dump(self.counter, F)