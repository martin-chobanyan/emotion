#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import warnings
from PIL import Image
from tqdm import tqdm
from google_images_download import google_images_download
from facenet_pytorch import MTCNN
from model import FaceFinder

warnings.filterwarnings("error")


# TODO: test both data collection methods again (especially with the UserWarning)
# ----------------------------------------------------------------------------------------------------------------------
#                                       Functions for collecting the emotion data
# ----------------------------------------------------------------------------------------------------------------------

def construct_query(emotion):
    """Maps the emotion to the optimal query

    The Google Images query for most emotion keywords will be "<emotion> human face".
    For a few exceptions, the optimal query is different.

    Parameters
    ----------
    emotion: str

    Returns
    -------
    query: str
    """
    exceptions = ['disgusted', 'grossed_out', 'resentful', 'elated']
    if emotion in exceptions:
        query = f'{emotion} face'
    else:
        query = f'{emotion} human face'
    return query


def filter_human_images(img_dir, face_finder):
    """Filter the results of an emotion query to RGB images that are uncorrupted and contain only one face

    Parameters
    ----------
    img_dir: str
        The directory containing the image results
    face_finder: FaceFinder
        The FaceFinder object used to detect the presence of a single human face

    Returns
    -------
    int, int, int
        The number of valid images, the number of images that did not meet the single face requirement, and the
        number of corrupted images
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


def download_emotions(emotion_map, data_dir, driver_dir, emotion_limit=2000, synonym_limit=1000):
    """Download the images for each emotion class

    This function uses the `google_images_download` package to run a Google Images query, retrieve the results,
    and store the images. The resulting images will be filtered to only those containing a single human face.

    The queries are define by the emotion map, which maps each main emotion class to a list of keywords that are
    synonymous. The images for the synonyms will be stored in the directory for their parent emotion class.

    Parameters
    ----------
    emotion_map: dict[str, list[str]]
        A dictionary mapping each emotion class to a list containing the emotion along with its synonyms.
    data_dir: str
        The root directory which will contain the emotion subdirectories.
        The subdirectories will contain the image results for the target emotion.
    driver_dir: str
        The path to the google chrome driver
    emotion_limit: int, optional
        The maximum number of images that can be returned for queries containing the main emotions (default=2000)
    synonym_limit: int, optional
        The maximum number of images that can be returned for queries containing the synonyms (default=1000).
        The synonyms should generally have a smaller limit because results near the end are more likely to deviate from
        the target emotion.
    """
    face_model = FaceFinder(MTCNN(keep_all=True))
    response = google_images_download.googleimagesdownload()
    args = {'keywords': '',
            'output_directory': data_dir,
            'image_directory': '',
            'silent_mode': True,
            'limit': -1,
            'chromedriver': driver_dir}
    for emotion_label, keywords in emotion_map.items():
        for k in keywords:
            search_limit = emotion_limit if (k == emotion_label) else synonym_limit
            args.update({'keywords': construct_query(k), 'image_directory': emotion_label, 'limit': search_limit})
            response.download(args)
        print(f'Filtering images for "{emotion_label}"')
        img_dir = os.path.join(data_dir, emotion_label)
        counts = filter_human_images(img_dir, face_model)
        print(f'Num of valid images:\t{counts[0]}')
        print(f'Num of non-human images removed:\t{counts[1]}')
        print(f'Num of corrupted images removed:\t{counts[2]}')
        print()


# ----------------------------------------------------------------------------------------------------------------------
#                                Functions for collecting the makeshift ImageNet data
# ----------------------------------------------------------------------------------------------------------------------

def filter_and_resize_images(img_dir, size=(224, 224)):
    """Filters the results to uncorrupted RGB images and then resizes them to the target size

    Parameters
    ----------
    img_dir: str
        The directory containing the image results
    size: tuple[int, int], optional
        The target image size (default=(224, 224))
    """
    for filename in tqdm(os.listdir(img_dir)):
        filepath = os.path.join(img_dir, filename)
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            img = img.resize(size)
            img.save(filepath)
        except:
            os.remove(filepath)


def download_fake_imagenet(labels, data_dir, driver_dir, label_limit=10):
    """Download images for each Imagenet label to mimic a smaller version of the dataset

    Parameters
    ----------
    labels: list[str]
        A list of target Imagenet labels
    data_dir: str
        The directory which will contain the label-subdirectories.
        The subdirectories will containing the image results for the target label.
    driver_dir: str
        The path to the chrome driver
    label_limit: int, optional
        The maximum number of images to return for each label (default=10)
    """
    response = google_images_download.googleimagesdownload()
    args = {'keywords': '',
            'output_directory': data_dir,
            'image_directory': '',
            'silent_mode': True,
            'limit': label_limit,
            'chromedriver': driver_dir}
    for label in labels:
        args.update({'keywords': label, 'image_directory': label})
        response.download(args)
        filter_and_resize_images(os.path.join(data_dir, label))


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    chrome_driver = '/home/mchobanyan/data/emotion/chromedriver'

    # download the emotion data
    data_dir = '/home/mchobanyan/data/emotion/images/imagenet/'
    emotions = {'angry': ['angry', 'furious', 'resentful', 'irate'],
                'disgusted': ['disgusted', 'sour', 'grossed out'],
                'happy': ['happy', 'smiling', 'cheerful', 'elated', 'joyful'],
                'sad': ['sad', 'depressed', 'sorrowful', 'mournful', 'grieving', 'crying'],
                'surprised': ['surprised', 'astonished', 'shocked', 'amazed']}
    download_emotions(emotions, data_dir, chrome_driver)

    # download the pseudo Imagenet data
    imagenet_labels = []
    with open('/home/mchobanyan/data/emotion/imagenet_labels.txt', 'r') as file:
        for line in file:
            imagenet_labels.append(line.strip())
    data_dir = '/home/mchobanyan/data/emotion/images/imagenet/'
    imagenet_label_file = '/home/mchobanyan/data/emotion/imagenet_labels.txt'
    download_fake_imagenet(imagenet_labels, data_dir, chrome_driver)
