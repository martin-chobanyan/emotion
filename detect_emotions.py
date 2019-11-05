#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file applies the trained emotion classifier to images and videos. It first extracts all faces from the
images/frames using the FaceFinder class. It then applies the necessary transformations (e.g. Grayscale) on the image
and passes it to the FacePredictor class (which wraps the capabilities of the emotion classifier to return the predicted
emotion label along with its softmax probability).
"""

from argparse import ArgumentParser
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from model import FaceFinder, FacePredictor, load_model
from utils import FontLoader

# define RGB color constants
RED = (200, 0, 0)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
PURPLE = (148, 0, 211)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)

# map each emotion to the color used for the bounding box
EMOTION_COLOR_MAP = {'angry': RED, 'happy': GREEN, 'surprised': BLUE, 'disgusted': PURPLE, 'sad': GRAY}


def process_image(face_images, boxes, predict, draw, font=None, font_loader=None, padding=5, line_width=3):
    """Draws a bounding box around each face in the image with the emotion label and confidence level

    Parameters
    ----------
    face_images: Image or list[Image]
        Either a single PIL Image of a face or a list of Image objects to process.
    boxes: np.ndarray
        A numpy array containing the bounding box extents for each face in face_images within the overall image.
        This argument is aligned with face_images. Bounding box is in (x_min, y_min, x_max, y_max) format.
        If face_images is a PIL Image then boxes should have only four elements.
        If face_images is a list of Image objects then boxes should have shape (n, 4)
    predict: nn.Module
        The emotion detector model that will be applied to each face image
    draw: ImageDraw
        The ImageDraw object for the image containing all of the faces. This object is modified in-place.
    font: ImageFont, optional
        The ImageFont object to use for the text annotation.
        If left as None, then the font_loader argument must be provided.
    font_loader: FontLoader, optional
        A FontLoader used to load the font given the width of each bounding box.
        If left as None, then the font argument must be provided.
    padding: int, optional
        The amount of pixel padding for the both the width and height of each bounding box (default=5)
    line_width: int, optional
        The pixel width of the bounding box (default=3)
    """
    if isinstance(face_images, Image.Image):
        face_images = [face_images]
        boxes = [boxes]

    for face_img, box in zip(face_images, boxes):
        emotion, prob = predict(face_img)
        prob = round(100 * prob, 2)
        message = f'{emotion}: {prob}%'
        color = EMOTION_COLOR_MAP[emotion]

        # draw the main bounding box outline for the face
        draw.rectangle(box.tolist(), outline=color, width=line_width)

        # get the face extents from the bounding box
        x_min, y_min, x_max, y_max = box

        if font_loader is not None:
            font = font_loader(x_max - x_min)

        # get the extents of the text message
        ascent, descent = font.getmetrics()
        text_width = font.getsize(message)[0]
        text_height = ascent + descent

        # draw the message over a filled-in background rectangle
        draw.rectangle((x_min, y_min, x_min + text_width + padding, y_min + text_height + padding), fill=color)
        draw.text((x_min + padding, y_min + padding), message, font=font, fill=WHITE)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, help='The path to the pickled model state')
    parser.add_argument('--image', required=True, help='The path to the target image')
    args = parser.parse_args()

    # load the final model that has been trained on the Google Images dataset
    model = load_model(args.model)

    # create the face finder using the pre-trained MTCNN model
    face_finder = FaceFinder(MTCNN(keep_all=True))

    # wrap the emotion classifier model in the FacePredictor
    predict_emotion = FacePredictor(model, emotions=sorted(list(EMOTION_COLOR_MAP.keys())))

    pil_img = Image.open(args.image).convert('RGB')
    faces, boxes = face_finder(pil_img, return_boxes=True)
    draw = ImageDraw.Draw(pil_img)

    process_image(faces, boxes, predict_emotion, draw, font_loader=FontLoader())
    pil_img.show()
