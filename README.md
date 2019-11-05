## Emotion detector
The goal of this project is to leverage existing, open-source tools to quickly train a deep learning model that detects emotions from human faces.
This repo offers tools to 
- scrape a dataset from Google Images and leverage the queries to label the images
- apply and fine-tune pretrained models to detect faces and serve as the starting point for the model
- use the trained model to detect and identify emotions in images and videos

The five target emotions that will be detected are:
1. **Angry**
2. **Disgusted**
3. **Happy**
4. **Sad**
5. **Surprised** 

For a detailed discussion of this project, see this accompanying [blog post](https://medium.com/swlh/training-an-emotion-detector-with-transfer-learning-91dea84adeed)

![labeled-friends.png](https://github.com/mcGIT123/emotion/blob/master/resources/labeled-friends.png)
![labeled-office.png](https://github.com/mcGIT123/emotion/blob/master/resources/labeled-office.png)

### Data Collection
The first step in training a supervised machine learning model is to collect labeled data.
A good starting point is the publicly available [FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data),
a collection of grayscale, 48x48 images of faces labeled by their expressed emotion. Once the CSV file is downloaded, 
**unpack_fer_data.py** can be used to extract, process, and store these images in a directory.

In addition to these images, **download_data.py** can be used to scrape data from Google Images and use the keywords to label the downloaded images. 
This is done using the 'google_images_download' library.

### Fine-tuning a pretrained ImageNet model
The base model for this project is a Resnet model pretrained on ImageNet (as is common for most computer vision transfer learning tasks).
This model will be fine-tuned later on in the pipeline to detect emotions from faces.

Before this is done, however, the pretrained model is altered to accept single-channel grayscale images.
It is then fine-tuned such that its output embedding for a grayscale image closely matches the output of the
vanilla model with the RGB version of the same image. 
This is done because later on in the pipeline, the faces will be transformed to grayscale before being fed to the model.
See **finetune_gray_imagenet.py** for this code.

The Google Images scraper can again be leveraged to pull down a small, "fake" sample of ImageNet by querying for a handful of images for each ImageNet label.
Again, see **download_data.py** for the code. See this [file](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) for all 1000 ImageNet labels.  

### Fine-tuning on the FER dataset
The next step is to fine-tune the model to detect emotions in the FER dataset. 
Note, the "Fear" and "Neutral" images were removed from this dataset in order to align with the images scraped from the internet.
See **finetune_fer.py** for this code.

### Fine-tuning on the scraped Google Images
In order to remove unnecessary features from the scraped data, the images are transformed into single-channel
grayscale images and are cropped such that each image contains only a single human face. 
The latter step is performed by applying a pretrained MTCNN face detection model to detect and return a bounding box of each face in an image.
This model comes from the 'facenet_pytorch' library. 

The code for the final training pipeline on the scraped images can be found in **finetune_emotion.py**.


### Applying the models
Most of the tools used to initialize, define, and train the models for this project can be found in **model.py**.
See **detect_emotions.py** for applying the trained model to detect and classify emotions across images and videos. 
    
   
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

### Installation
See [this page](https://google-images-download.readthedocs.io/en/latest/installation.html)
for details on how to install the google_images_download library.
In particular, 'selenium' and a Google Chrome driver must be [downloaded](https://google-images-download.readthedocs.io/en/latest/troubleshooting.html#installing-the-chromedriver-with-selenium) in order to query for more images.
 
