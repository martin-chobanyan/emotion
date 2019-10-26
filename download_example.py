#!/usr/bin/env python
# -*- coding: utf-8 -*-

from google_images_download import google_images_download

DRIVER_PATH = 'insert the filepath to the chrome driver here...'
args = {'keywords': 'happy human face',  # the query to search in Google Images
        'output_directory': '/data/emotion/images/',  # the root directory for the images
        'image_directory': 'happy',  # the subdirectory (images will be downloaded in '/data/emotion/images/happy/')
        'silent_mode': True,  # limits output printed to stdout
        'limit': 1000,  # the limit for the number of images returned
        'chromedriver': DRIVER_PATH}
       
response = google_images_download.googleimagesdownload()
response.download(args)
