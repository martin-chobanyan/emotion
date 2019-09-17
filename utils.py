#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# TODO: move this to data science kit
def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues, figsize=(6, 4)):
    if not title:
        title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def calc_norm_stats(data, num_channels=3):
    """Dataset must return tensors with shape [num_channels, height, width]"""
    n = len(data)
    means = np.zeros(num_channels)
    stdvs = np.zeros(num_channels)
    for x, _ in tqdm(data):
        for i in range(num_channels):
            means[i] += x[i, :, :].mean().item()
            stdvs[i] += x[i, :, :].std().item()
    means /= n
    stdvs /= n
    return means.tolist(), stdvs.tolist()


class RecoverImage:
    def __init__(self, means=None, stdvs=None):
        self.means = means
        self.stdvs = stdvs

    def __call__(self, x):
        if self.means is not None:
            means = torch.Tensor(self.means).view(3, 1, 1)
            stdvs = torch.Tensor(self.stdvs).view(3, 1, 1)
            x = (x * stdvs) + means
        x *= 255
        x = x.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(x)
