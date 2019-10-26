#!/usr/bin/env python
# -*- coding: utf-8 -*-

from google_images_download import google_images_download

DRIVER_PATH = 'insert the filepath to the chrome driver here...'
args = {'keywords': 'happy human face',
        'output_directory': '/data/emotion/images/',
        'image_directory': 'happy',
        'silent_mode': True,
        'limit': 1000,
        'chromedriver': DRIVER_PATH}
       
response = google_images_download.googleimagesdownload()
response.download(args)
