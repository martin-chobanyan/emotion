## Emotion detector

This repo provides an easy pipeline for creating an facial, emotion detector from close to scratch. It includes the following components:
- **Data collection and processing**
    - download_data.py uses a Google Image scraping library to collect thousands of facial images labeled by the keyword search
    - unpack_fer_data.py unpacks the FER data from the Kaggle CSV file into a directory of 48x48 grayscale facial images.
- **Gray ImageNet finetuning**
    - finetune_gray_imagenet.py
- **FER data finetuning**
    - finetune_fer.py
- **Google Image data finetuning**
    - finetune_emotion.py
- **Applying the emotion detector to images and videos**
    - detect_emotions.py
- **Other files**
    - model.py
    - utils.py
    
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
