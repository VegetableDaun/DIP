import numpy as np
import os
import json

from tensorflow import keras
from PIL import Image
from pathlib import Path


class CustomCallback(keras.callbacks.Callback):
    # path=Path().resolve()
    def __init__(self, num_save=5, path=Path('/content/drive/Othercomputers/Ноутбук/3/Code'), noise='noise.json'):
        super(CustomCallback, self).__init__()
        self.num_save = num_save
        self.path = path

        with open(self.path / noise, mode='r') as F:
            self.noise = np.array(json.load(F))

    def on_train_begin(self, logs=None):
        if os.path.isfile(self.path / 'train_models/metrics.json'):
            with open(self.path / 'train_models/metrics.json', mode='r') as F:
                feeds = dict(json.load(F))
                completed_steps = max(map(int, feeds.keys()))
                while (completed_steps % self.num_save != 0):
                    del feeds[str(completed_steps)]
                    completed_steps -= 1
            if completed_steps != 0:
                with open(self.path / 'train_models/metrics.json', mode='w') as F:
                    json.dump(feeds, F)
            else:
                os.remove(self.path / 'train_models/metrics.json')

    def on_epoch_end(self, epoch, logs=None):
        add_value = {}
        if not os.path.isfile(self.path / 'train_models/metrics.json'):
            with open(self.path / 'train_models/metrics.json', mode='w') as F:
                add_value[1] = logs
                json.dump(add_value, F)
        else:
            with open(self.path / 'train_models/metrics.json', mode='r') as F:
                feeds = dict(json.load(F))
                feeds_keys = map(int, feeds.keys())
                add_key = max(feeds_keys) + 1
                feeds[add_key] = logs
            with open(self.path / 'train_models/metrics.json', mode='w') as F:
                json.dump(feeds, F)

        if (epoch + 1) % self.num_save == 0:
            self.model.discriminator.save(self.path / 'train_models/Discriminator.hdf5')
            self.model.generator.save(self.path / 'train_models/Generator.hdf5')

            img = np.array(self.model.generator(self.noise))[0]
            img = np.round(img * 255)
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(self.path / ('train_models/Image/Epoch_' + str(add_key) + '.jpg'))
