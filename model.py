#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor


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
    def __call__(self, img, return_boxes=False):
        """Extract the image region containing the single face

        Parameters
        ----------
        img: Image

        Returns
        -------
        Image or list[Image]
            If there is one face, then a single Image is returned. If multiple faces are found, then a list of Image
            objects are returned.
        """
        boxes, _ = self.face_model.detect(img)
        if boxes is not None:
            if len(boxes) == 1:
                face_img = img.crop(boxes[0])
                if return_boxes:
                    return face_img, boxes[0]
                return face_img

            faces = [img.crop(box) for box in boxes]
            if return_boxes:
                return faces, boxes
            return faces
        return None


# if the prediction is below some %, then output 'neutral...'
class FacePredictor:
    def __init__(self, model, emotions, img_size=(224, 224)):
        self.model = model.eval()
        self.emotions = emotions
        self.transform = Compose([Resize(img_size), Grayscale(1), ToTensor()])

    def __call__(self, face):
        x = self.transform(face)
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))
            probs = F.softmax(out, dim=1).squeeze()
            idx = torch.argmax(probs)
            return self.emotions[idx], probs[idx].item()


def init_grayscale_resnet():
    gray_model = resnet50(pretrained=True)
    w = torch.zeros((64, 1, 7, 7))
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    gray_model.conv1.weight.data = w
    return gray_model


def load_model(model_state, num_emotions=5):
    model = init_grayscale_resnet()
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, num_emotions)
    model.load_state_dict(torch.load(model_state))
    return model


def predict_label(output):
    return torch.argmax(F.softmax(output, dim=1)).item()


def accuracy(model_out, true_labels):
    pred = torch.argmax(F.softmax(model_out, dim=1), dim=1)
    acc = (pred == true_labels.squeeze())
    return float(acc.sum()) / len(acc)


def train_epoch(model, dataloader, criterion, optimizer, device):
    avg_loss = []
    model.train()
    for batch_image, batch_label in dataloader:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)
        optimizer.zero_grad()
        output = model(batch_image)
        loss = criterion(output, batch_label)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
    return sum(avg_loss) / len(avg_loss)


def val_epoch(model, dataloader, criterion, device):
    avg_loss = []
    avg_acc = []
    model.eval()
    with torch.no_grad():
        for batch_image, batch_label in dataloader:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)
            output = model(batch_image)
            loss = criterion(output, batch_label)
            acc = accuracy(output, batch_label)
            avg_loss.append(loss.item())
            avg_acc.append(acc)
    return sum(avg_loss) / len(avg_loss), sum(avg_acc) / len(avg_acc)
