#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont

MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 100
MAX_MSG_SIZE = len('surprised: 100.00%')


class FontLoader:
    """Load a PIL ImageFont given the width of the bounding box

    This class provides a mapping from the width of an image bounding box to the largest ImageFont that does not exceed
    the extents of the bounding box (except for very small bounding boxes with only a few pixels).

    Note: As of now only the FreeMonoBold font is supported

    Parameters
    ----------
    min_size: int, optional
        The min font size to consider (default=8)
    max_size: int, optional
        The maximum font size to consider (default=50)
    max_msg_size: int, optional
        The maximum message size to consider (default=18; length of 'surprised: 100.00%')
    """

    def __init__(self, min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, max_msg_size=MAX_MSG_SIZE):
        num_fonts = max_size - min_size + 1
        self.font_sizes = np.zeros(num_fonts, dtype=np.int32)
        self.msg_widths = np.zeros(num_fonts, dtype=np.int32)

        for i in range(num_fonts):
            font_size = min_size + i
            message = ' ' * max_msg_size
            font = self.get_font(font_size)
            message_width = font.getsize(message)[0]
            self.font_sizes[i] = font_size
            self.msg_widths[i] = message_width

        self.min_size = min_size
        self.max_size = max_size
        self.max_msg_size = max_msg_size

    @staticmethod
    def get_font(font_size):
        """A static method that returns the appropriate ImageFont given the font size

        Parameters
        ----------
        font_size: int

        Returns
        -------
        ImageFont.FreeTypeFont
        """
        return ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', font_size)

    def __call__(self, img_width):
        """Return the largest ImageFont that does not exceed the image width using the maximum message length

        Parameters
        ----------
        img_width: int
            The width of the image bounding box in pixels

        Returns
        -------
        ImageFont.FreeFontType
        """
        nearest_width_idx, = np.where(self.msg_widths < img_width)
        if len(nearest_width_idx) > 0:
            idx = nearest_width_idx[-1]
        else:
            idx = 0
        font_size = self.font_sizes[idx]
        return self.get_font(font_size)


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
            means = torch.tensor(self.means).view(3, 1, 1)
            stdvs = torch.tensor(self.stdvs).view(3, 1, 1)
            x = (x * stdvs) + means
        x *= 255
        x = x.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(x)


def resize_and_save(dataset, out_dir):
    """Crop each image from the dataset and store in the output directory

    Parameters
    ----------
    dataset: Dataset
        The pytorch Dataset class for generating the images
    out_dir: str
        The path to the output directory
    """
    labels = dataset.classes
    for i, (img, label_idx) in enumerate(tqdm(dataset)):
        emotion = labels[label_idx]
        output_dir = os.path.join(out_dir, emotion)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img.save(os.path.join(output_dir, f'{emotion}_{i}.png'), 'PNG')


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
