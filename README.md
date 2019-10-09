## Emotion detector
The goal of this project is to leverage existing, open-source tools to quickly train a deep learning model that detects emotions from human faces.
This repo offers tools to 
- scrape a dataset from Google Images and leverage the queries to label the images
- apply and fine-tune pretrained models to detect faces and serve as the starting point for the model
- use the trained model to detect and identify emotions in images and videos

To go into more detail about the project, see this blog post (add the link when it is done...)  

### Data Collection
The first step in training a supervised machine learning model is to collect labeled data.
A good starting point is the publicly available [FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data),
a collection of grayscale, 48x48 images of faces labeled by their expressed emotion. Once the CSV file is downloaded, 
**unpack_fer_data.py** can be used to extract, process, and store these images in a directory.

In addition to these images, **download_data.py** can be used to scrape data from Google Images and use the keywords to label the downloaded images. 
This is done using the 'google_images_download' python package.

### Fine-tuning a pretrained ImageNet model
The base model for this project is a Resnet model pretrained on ImageNet (as is common for most computer vision transfer learning tasks).
This model will be fine-tuned later on in the pipeline to detect emotions from faces.

Before this can be done, however, the pretrained model is altered to accept single-channel grayscale images.
It is then fine-tuned such that its output embedding for a grayscale image closely matches the output of the
vanilla model with the RGB version of the same image. 
This is done because later on in the pipeline, the faces will be transformed to grayscale before being fed to the model.

The Google Images scraper can again be leveraged to pull down a small, "fake" sample of ImageNet by querying for a handful of images for each ImageNet label.
Again, see **download_data.py** for the code. See this [file](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) for all 1000 ImageNet labels.  

### Fine-tuning on the FER dataset


### Fine-tuning on the scraped Google Images



### Applying the models
Most of the tools used to initialize, train, and apply the models for this project can be found in **model.py**.
    
### Requirements
```
google_images_download
facenet_pytorch
numpy
pandas
pillow
torch
torchvision
tqdm
```
