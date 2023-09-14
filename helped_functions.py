import cv2
import numpy as np
from keras.utils import to_categorical
import pandas as pd


def conv_RGB(img, label):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))

    if img.shape[2] == 1:
        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = to_categorical(label, num_classes=3)

    return img, label


def increase(data, add_labels):
    add = pd.DataFrame()
    # if Data.iloc[i]['label'] in add_labels:
    for i in range(len(data)):
        for _ in range(add_labels[data.iloc[i]['label']]):
            add = pd.concat([add, data.iloc[[i]]])

    add = add.replace({'add': 0}, 1)
    data = pd.concat([data, add]).reset_index(drop=True)

    return data
