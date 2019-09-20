#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file applies the trained emotion classifier to images and videos. It first extracts all faces from the
images/frames using the FaceFinder class. It then applies the necessary transformations (e.g. Grayscale) on the image
and passes it to the FacePredictor class (which wraps the capabilities of the emotion classifier to return the predicted
emotion label along with its softmax probability).
"""

from model import FaceFinder, FacePredictor