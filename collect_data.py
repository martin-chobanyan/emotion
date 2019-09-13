#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 'scared': ['scared', 'fearful', 'afraid', 'anxious', 'panic'],

import os
import warnings
from PIL import Image
from tqdm import tqdm
from google_images_download import google_images_download
from facenet_pytorch import MTCNN
from check_face import FaceFinder

warnings.filterwarnings("error")

# TODO: test both data collection methods again (especially with the UserWarning)

# EMOTIONS


def construct_query(emotion):
    exceptions = ['disgusted', 'grossed_out', 'resentful', 'elated']
    if emotion in exceptions:
        query = f'{emotion} face'
    else:
        query = f'{emotion} human face'
    return query


def filter_human_images(img_dir, face_finder):
    """
    Parameters
    ----------
    img_dir: str
    face_finder: FaceFinder
    """
    num_human = 0
    num_inhuman = 0
    num_corrupted = 0
    for filename in tqdm(os.listdir(img_dir)):
        try:
            img = Image.open(os.path.join(img_dir, filename))
            img = img.convert('RGB')
            if face_finder.has_single_face(img):
                num_human += 1
            else:
                num_inhuman += 1
                os.remove(os.path.join(img_dir, filename))
        except:
            os.remove(os.path.join(img_dir, filename))
            num_corrupted += 1
    return num_human, num_inhuman, num_corrupted


def download_emotions(data_dir, driver_dir):
    emotion_map = {'angry': ['angry', 'furious', 'resentful', 'irate'],
                   'disgusted': ['disgusted', 'sour', 'grossed out'],
                   'happy': ['happy', 'smiling', 'cheerful', 'elated', 'joyful'],
                   'sad': ['sad', 'depressed', 'sorrowful', 'mournful', 'grieving', 'crying'],
                   'surprised': ['surprised', 'astonished', 'shocked', 'amazed']}

    face_model = FaceFinder(MTCNN(keep_all=True))

    # instantiate the downloader
    response = google_images_download.googleimagesdownload()

    # set up the argument template
    args = {'keywords': '',
            'output_directory': data_dir,
            'image_directory': '',
            'silent_mode': True,
            'limit': -1,
            'chromedriver': driver_dir}

    for emotion_label, keywords in emotion_map.items():
        for k in keywords:
            search_limit = 2000 if (k == emotion_label) else 1000
            args.update({'keywords': construct_query(k), 'image_directory': emotion_label, 'limit': search_limit})
            response.download(args)
        print(f'Filtering images for "{emotion_label}"')
        img_dir = os.path.join(data_dir, emotion_label)
        counts = filter_human_images(img_dir, face_model)
        print(f'Num of valid images:\t{counts[0]}')
        print(f'Num of non-human images removed:\t{counts[1]}')
        print(f'Num of corrupted images removed:\t{counts[2]}')
        print()


# IMAGENET


def filter_and_resize_images(img_dir):
    for filename in tqdm(os.listdir(img_dir)):
        filepath = os.path.join(img_dir, filename)
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img.save(filepath)
        except:
            os.remove(filepath)


def download_fake_imagenet(data_dir, driver_dir, label_file):
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            labels.append(line.strip())

    response = google_images_download.googleimagesdownload()
    args = {'keywords': '',
            'output_directory': data_dir,
            'image_directory': '',
            'silent_mode': True,
            'limit': 10,
            'chromedriver': driver_dir}

    for label in labels:
        args.update({'keywords': label, 'image_directory': label})
        response.download(args)
        filter_and_resize_images(os.path.join(data_dir, label))


if __name__ == '__main__':
    data_dir = '/home/mchobanyan/data/emotion/images/imagenet/'
    chrome_driver = '/home/mchobanyan/data/emotion/chromedriver'
    imagenet_label_file = '/home/mchobanyan/data/emotion/imagenet_labels.txt'
    # download_emotions(data_dir, chrome_driver)
    download_fake_imagenet(data_dir, chrome_driver, imagenet_label_file)
