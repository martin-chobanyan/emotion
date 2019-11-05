#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor


# ----------------------------------------------------------------------------------------------------------------------
#                                       Models operating on images of faces
# ----------------------------------------------------------------------------------------------------------------------

class FaceFinder:
    """This class wraps the capabilities of face detection models to detect and extract faces from images

    Parameters
    ----------
    model: nn.Module
        A pytorch module with a `detect` method that returns a list of bounding boxes for detected face along with a
        list of probabilities for each bounding box (the latter is ignored here). All models from facenet_pytorch
        include such a method.
    """

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

    def __call__(self, img, return_boxes=False):
        """Extract the image region containing the single face

        Parameters
        ----------
        img: Image
        return_boxes: bool, optional
            If True, then a the bounding box of each face is returned as a numpy array. The format of the bounding boxes
            are (x_min, y_min, x_max, y_max). If a single face is detected, then the array will have four elements.
            If multiple faces are detected then the array will have shape (n, 4). Default is False.

        Returns
        -------
        Image or list[Image]
            If there is one face, then a single Image is returned. If multiple faces are found, then a list of Image
            objects are returned. If return_boxes is True, then the bounding boxes of the faces are returned as well.
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


class FacePredictor:
    """Predict the emotion of a detected face

    Parameters
    ----------
    model: nn.Module
        An pytorch model that classifies the emotion expressed in an image of a face
    emotions: list[str]
        A list of the emotions which is aligned with the output vector of the model
    img_size: tuple[int, int], optional
        The expected (width, height) dimensions for the model input.
        All images will be resized to this dimension (default=(224, 224))
    device: torch.device, optional
        The device to run the model on. Will run on the CPU if None (default=None)
    """

    def __init__(self, model, emotions, img_size=(224, 224), device=None):
        self.emotions = emotions
        self.transform = Compose([Resize(img_size), Grayscale(1), ToTensor()])
        self.device = torch.device('cpu') if device is None else device
        self.model = model.eval().to(self.device)

    def __call__(self, face):
        """Classify the expressed emotion

        Parameters
        ----------
        face: PIL.Image.Image
            An image of a face (should be zoomed in to just the face).

        Returns
        -------
        str, float
            Returns the predicted emotion along with the model's confidence
        """
        x = self.transform(face).to(self.device)
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))
            probs = F.softmax(out, dim=1).squeeze()
            idx = torch.argmax(probs).item()
            return self.emotions[idx], probs[idx].item()


# ----------------------------------------------------------------------------------------------------------------------
#                               Utilities for initializing, loading, and saving models
# ----------------------------------------------------------------------------------------------------------------------


def init_grayscale_resnet(init_weights=None):
    """Initialize a pretrained Resnet-50 model and change the first layer to be a one-channel 2D convolution

    Returns
    -------
    gray_model: nn.Module
    """
    gray_model = resnet50(pretrained=True)
    if init_weights is None:
        w = torch.zeros((64, 1, 7, 7))
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    else:
        w = init_weights
    gray_model.conv1.weight.data = w
    return gray_model


def load_model(model_state, num_emotions=5):
    """Redefine the model architecture and load the parameter state from a saved model

    Parameters
    ----------
    model_state: str
        The path to the saved model state dictionary
    num_emotions: int, optional
        The number of emotions to classify (default=5)

    Returns
    -------
    model: nn.Module
    """
    model = init_grayscale_resnet()
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, num_emotions)
    model.load_state_dict(torch.load(model_state))
    return model


def checkpoint(model, filepath):
    """Save the state of the model

    To restore the model do the following:
    >> the_model = TheModelClass(*args, **kwargs)
    >> the_model.load_state_dict(torch.load(PATH))

    Parameters
    ----------
    model: nn.Module
        The pytorch model to be saved
    filepath: str
        The filepath of the pickle
    """
    torch.save(model.state_dict(), filepath)


# ----------------------------------------------------------------------------------------------------------------------
#                               Functions for training the emotion classifier model
# ----------------------------------------------------------------------------------------------------------------------


def predict_label(output):
    """Get the index of the emotion with the highest softmax probability

    Parameters
    ----------
    output: torch.Tensor
        The output of the emotion classifier (for one example, shape: [num emotions])

    Returns
    -------
    int
    """
    return torch.argmax(F.softmax(output, dim=1)).item()


def accuracy(model_out, true_labels):
    """Calculate the accuracy of a batch of predictions

    Parameters
    ----------
    model_out: torch.FloatTensor
        The output of the emotion classifier with shape [batch size, num emotions]
    true_labels: torch.LongTensor
        The true encoded emotion labels aligned with the model's output

    Returns
    -------
    float
    """
    pred = torch.argmax(F.softmax(model_out, dim=1), dim=1)
    acc = (pred == true_labels.squeeze())
    return float(acc.sum()) / acc.size(0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for an epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    float
        The average loss
    """
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
    """Run the model for a validation epoch

    Parameters
    ----------
    model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    device: torch.device

    Returns
    -------
    float, float
        The average loss and the average accuracy
    """
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
