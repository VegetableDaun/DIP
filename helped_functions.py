import cv2
import numpy as np
from keras.utils import to_categorical
import pandas as pd


def Conv(img, label):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))

    if img.shape[2] == 1:
        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = to_categorical(label, num_classes=3)

    return img, label


def increase(Data, add_labels):
    add = pd.DataFrame()
    # if Data.iloc[i]['label'] in add_labels:
    for i in range(len(Data)):
        for _ in range(add_labels[Data.iloc[i]['label']]):
            add = pd.concat([add, Data.iloc[[i]]])

    add = add.replace({'add': 0}, 1)
    Data = pd.concat([Data, add]).reset_index(drop=True)

    return Data
