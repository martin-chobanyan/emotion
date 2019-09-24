#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file applies the trained emotion classifier to images and videos. It first extracts all faces from the
images/frames using the FaceFinder class. It then applies the necessary transformations (e.g. Grayscale) on the image
and passes it to the FacePredictor class (which wraps the capabilities of the emotion classifier to return the predicted
emotion label along with its softmax probability).
"""

import os
import torch
from facenet_pytorch import MTCNN
from model import FaceFinder, FacePredictor, load_model

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    EMOTIONS = ['angry', 'disgusted', 'happy', 'sad', 'surprised']
    MODEL_PATH = '/home/mchobanyan/data/emotion/models/emotion_detect/gray-base/model_10.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH)

    face_finder = FaceFinder(MTCNN(keep_all=True))
    predict_emotion = FacePredictor(model, EMOTIONS)

    
