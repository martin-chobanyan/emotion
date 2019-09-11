#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm

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
