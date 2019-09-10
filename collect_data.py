#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from PIL import Image
from tqdm import tqdm
from google_images_download import google_images_download
from facenet_pytorch import MTCNN
from check_face import FaceFinder


def crop_and_save(dataset, rootdir, subdir):
    labels = dataset.classes
    for i, (img, label_idx) in enumerate(tqdm(dataset)):
        emotion = labels[label_idx]
        output_dir = os.path.join(rootdir, subdir, emotion)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img.save(os.path.join(output_dir, f'{emotion}_{i}.png'), 'PNG')


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
            if face_finder.has_single_face(img):
                num_human += 1
            else:
                num_inhuman += 1
                os.remove(os.path.join(img_dir, filename))
        except:
            os.remove(os.path.join(img_dir, filename))
            num_corrupted += 1
    return num_human, num_inhuman, num_corrupted


if __name__ == '__main__':
    # define the emotion labels
    emotion_map = {'angry': ['angry', 'furious', 'resentful', 'irate'],
                   'disgusted': ['disgusted', 'sour', 'grossed out'],
                   'happy': ['happy', 'smiling', 'cheerful', 'elated', 'joyful'],
                   'sad': ['sad', 'depressed', 'sorrowful', 'mournful', 'grieving', 'crying'],
                   # 'scared': ['scared', 'fearful', 'afraid', 'anxious', 'panic'],
                   'surprised': ['surprised', 'astonished', 'shocked', 'amazed']}

    # define the file paths
    data_dir = '/home/mchobanyan/data/emotion/images/'

    # finds human faces
    face_model = FaceFinder(MTCNN(keep_all=True))

    # instantiate the downloader
    response = google_images_download.googleimagesdownload()

    # set up the argument template
    args = {'keywords': '',
            'output_directory': data_dir,
            'image_directory': '',
            'silent_mode': True,
            'limit': -1,
            'chromedriver': '/home/mchobanyan/data/emotion/chromedriver'}

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
