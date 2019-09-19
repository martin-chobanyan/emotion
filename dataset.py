#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Resize
from facenet_pytorch import MTCNN
from model import FaceFinder


if __name__ == '__main__':

