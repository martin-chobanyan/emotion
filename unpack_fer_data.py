#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file processes the raw FER data from kaggle:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

The data comes as a CSV file with each row containing the emotion label and the image as a
flattened array of 2304 pixels. The images are extracted, resized, and stored in an output directory.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image


RAW_SHAPE = (48, 48)  # the original FER images are 48x48
TARGET_SHAPE = (224, 224)


def pixels_to_img(pixels):
    pixels = pixels.split(' ')
    img = np.array([int(pix) for pix in pixels])
    img = img.reshape(RAW_SHAPE).astype(np.uint8)
    return Image.fromarray(img)


if __name__ == '__main__':
    # define a mapping from the encoded label to the respective emotion (excluding "fear" and "neutral")
    label_map = {0: 'angry', 1: 'disgust', 3: 'happy', 4: 'sad', 5: 'surprised'}

    root_dir = '/home/mchobanyan/data/emotion/images/fer2013/'
    data_file = os.path.join(root_dir, 'fer2013.csv')
    out_dir = os.path.join(root_dir, 'fer')

    df = pd.read_csv(data_file)

    # drop emotions 2 and 6 (Fear and Neutral)
    mask = (df['emotion'] != 2) & (df['emotion'] != 6)
    df = df.loc[mask]
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        label, pixels = row[['emotion', 'pixels']]
        img = pixels_to_img(pixels)
        emotion = label_map[label]

        img_dir = os.path.join(out_dir, emotion)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        img = img.resize(TARGET_SHAPE)
        img.save(os.path.join(img_dir, f'img_{i}.png'))
