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

    # TODO: limit code reuse
    def __call__(self, img):
        """Extract the image region containing the single face

        Parameters
        ----------
        img: Image

        Returns
        -------
        Image
        """
        if self.has_single_face(img):
            boxes, _ = self.face_model.detect(img)
            return img.crop(boxes[0])
