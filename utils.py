#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


class RecoverImage:
    """Recover the original image after the ToTensor and Normalize transformations have been applied

    Parameters
    ----------
    means: list[float], optional
        A list of per channel means. If None, then it is assumed that Normalize was not used (default=None).
    stdvs: list[float], optional
        A list of per channel standard deviations.
        If None, then it is assumed that Normalize was not used (default=None).
    """

    def __init__(self, means=None, stdvs=None):
        self.means = means
        self.stdvs = stdvs

    def __call__(self, x):
        """Recover the image

        Parameters
        ----------
        x: torch.Tensor
            The transformed image tensor

        Returns
        -------
        PIL.Image.Image
            The recovered image as a PIL Image
        """
        if self.means is not None:
            means = torch.Tensor(self.means).view(3, 1, 1)
            stdvs = torch.Tensor(self.stdvs).view(3, 1, 1)
            x = (x * stdvs) + means
        x *= 255
        x = x.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(x)


def calc_norm_stats(data, num_channels=3):
    """Estimates the per channel means and standard deviations for a Dataset

    Parameters
    ----------
    data: Dataset
    num_channels: int, optional
        The number of channels in each image (default=3)

    Returns
    -------
    list[float], list[float]
        A list of per channel means and a list of per channel standard deviations
    """
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
