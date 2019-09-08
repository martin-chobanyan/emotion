#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2 as cv
from PIL import Image, ImageDraw


# UPDATED VERSION: using facenet_pytorch
class FaceFinder:
    def __init__(self, model):
        self.face_model = model

    def has_single_face(self, img):
        """
        Parameters
        ----------
        img: Image

        Returns
        -------
        bool
        """
        boxes, _ = self.face_model.detect(img)
        return (boxes is not None) and (len(boxes) == 1)
