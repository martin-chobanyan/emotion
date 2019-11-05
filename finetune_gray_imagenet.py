#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file defines a script that finetunes a modified version of an Imagenet-pretrained Resnet model.
Since the input to the emotion classifier will be transformed to grayscale, using the standard pretrained Resnet
will not be optimal (since it was trained on RGB images).

Instead, a new version of the pretrained model is created by swapping the first convolutional layer with a new,
initialized, 1-channel convolution layer. This layer alone is then finetuned to match
the output of the RGB pretrained Resnet model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor
from model import init_grayscale_resnet, checkpoint
from utils import RecoverImage

# imagenet color normalization
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDVS = [0.229, 0.224, 0.225]


# TODO: try adding an extra preprocessing layer that maps a 1-channel grayscale image to the 3-channel Resnet input

# ----------------------------------------------------------------------------------------------------------------------
#                                         Tools for fine-tuning the gray model
# ----------------------------------------------------------------------------------------------------------------------


class ColorAndGrayImages(ImageFolder):
    """Retrieves and returns an image along with its grayscale version

    This class extends the behavior of ImageFolder, meaning images should be arranged in subdirectories where then name
    of the subdirectory is the label. Along with returning the standard RGB version of the image, this class will also
    return the one-channel grayscale version.

    Parameters
    ----------
    image_dir: str
        The root directory containing the label subdirectories
    colored_transform: callable, optional
        An transformation that can be applied to the color image (default=None)
    gray_transform: callable, optional
        An transformation that can be applied to the grayscale image (default=None)
    """

    def __init__(self, image_dir, colored_transform=None, gray_transform=None):
        super().__init__(image_dir)
        self.colored_transform = colored_transform
        self.gray_transform = gray_transform

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        gray_img = Grayscale(1)(img)
        if self.colored_transform is not None:
            img = self.colored_transform(img)
        if self.gray_transform is not None:
            gray_img = self.gray_transform(gray_img)
        return img, gray_img


def train_epoch_gray(gray_model, color_model, dataloader, criterion, optimizer, device):
    """Trains the grayscale model by minimizing the loss between the gray model's output and the color model's output

    Note that the only trainable portion of the gray model should be the first convolution layer.
    The color model should not be trainable at all (it is only used to create the target output for the gray model).

    Parameters
    ----------
    gray_model: nn.Module
    color_model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    float
        The average loss across the epoch
    """
    avg_loss = []
    gray_model.train()
    color_model.train()
    for color_batch, gray_batch in dataloader:
        color_batch = color_batch.to(device)
        gray_batch = gray_batch.to(device)

        optimizer.zero_grad()
        output1 = gray_model(gray_batch)
        output2 = color_model(color_batch)

        loss = criterion(output1, output2)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

    return sum(avg_loss) / len(avg_loss)


def val_epoch_gray(gray_model, color_model, dataloader, criterion, device):
    """Runs a validation epoch using the grayscale model and the color model

    Parameters
    ----------
    gray_model: nn.Module
    color_model: nn.Module
    dataloader: DataLoader
    criterion: callable loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    float
        The average loss across the epoch
    """
    avg_loss = []
    gray_model.eval()
    color_model.eval()
    with torch.no_grad():
        for color_batch, gray_batch in dataloader:
            color_batch = color_batch.to(device)
            gray_batch = gray_batch.to(device)

            output1 = gray_model(gray_batch)
            output2 = color_model(color_batch)

            loss = criterion(output1, output2)
            avg_loss.append(loss.item())

    return sum(avg_loss) / len(avg_loss)


# ----------------------------------------------------------------------------------------------------------------------
#                                            Tools for comparing the models
# ----------------------------------------------------------------------------------------------------------------------


def top_k(output, k):
    """Returns the top-k predicted label index along with their associated probabilities

    Parameters
    ----------
    output: torch.Tensor
        The model output
    k: int

    Returns
    -------
    torch.LongTensor, torch.FloatTensor
         The top-k label indices and their associated probabilities
    """
    probs = F.softmax(output, dim=1).squeeze()
    pred_idx = torch.argsort(probs, descending=True)[:k]
    return pred_idx, probs[pred_idx]


class CompareModels:
    """Compare the model (color) Imagenet-pretrained model with the grayscale variant

    Parameters
    ----------
    gray_model: nn.Module
    color_model: nn.Module
    labels: list[str]
        A list of the 1000 Imagenet labels in order
    means: list[float], optional
        The per channel means used to normalize the RGB images (default=None; uses Imagenet means)
    stdvs: list[float], optional
        The per channel standard deviations used to normalize the RGB images
        (default=None; uses Imagenet standard deviations)
    """

    def __init__(self, gray_model, color_model, labels, means=None, stdvs=None):
        self.gray_model = gray_model.eval()
        self.color_model = color_model.eval()
        self.labels = labels
        self.recover = RecoverImage(IMAGENET_MEANS if means is None else means,
                                    IMAGENET_STDVS if means is None else stdvs)

    def __call__(self, color_tensor, gray_tensor, k=5):
        """Run each model on the inputs and print the top-k predicted labels along with their confidences

        Parameters
        ----------
        color_tensor: torch.Tensor
            The tensor input for the color model
        gray_tensor: torch.Tensor
            The tensor input for the gray model
        k: int, optional
            The k value for the finding the top-k predictions
        """
        with torch.no_grad():
            color_pred = self.color_model(color_tensor.unsqueeze(0))
            pred_idx, probs = top_k(color_pred, k)
            print('Color predictions:')
            for idx, p in zip(pred_idx.tolist(), probs.tolist()):
                print(f'{self.labels[idx]}: {round(100 * p, 2)}%')
            print()

            gray_pred = self.gray_model(gray_tensor.unsqueeze(0))
            pred_idx, probs = top_k(gray_pred, k)
            print('Gray predictions:')
            for idx, p in zip(pred_idx.tolist(), probs.tolist()):
                print(f'{self.labels[idx]}: {round(100 * p, 2)}%')

        color_img = self.recover(color_tensor)
        return color_img


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    P_TRAIN = 0.8  # proportion of examples to use for training
    BATCH_SIZE = 32
    NUM_WORKERS = 6
    NUM_EPOCHS = 50
    learning_rate = 0.001
    DECAY_RATE = 5  # number of epochs after which to decay the learning rate
    LR_DECAY = 0.5  # amount to decrease the learning rate every 'DECAY_RATE' epochs
    CHECKPOINT_RATE = 5  # number of epochs after which to checkpoint the model
    IMAGE_DIR = '/home/mchobanyan/data/emotion/images/imagenet/'
    MODEL_DIR = '/home/mchobanyan/data/emotion/models/emotion_detect/imagenet/'

    dataset = ColorAndGrayImages(image_dir=IMAGE_DIR,
                                 colored_transform=Compose([ToTensor(), Normalize(IMAGENET_MEANS, IMAGENET_STDVS)]),
                                 gray_transform=ToTensor())
    print(f'Number of images: {len(dataset)}')

    train_size = int(len(dataset) * P_TRAIN)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # initialize both models
    color_model = resnet50(pretrained=True)
    gray_model = init_grayscale_resnet()

    # only allow the first convolution layer in the gray model to train
    for param in gray_model.parameters():
        param.requires_grad = False
    gray_model.conv1.weight.requires_grad = True

    # prepare each model for the fine-tuning
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    color_model = color_model.to(torch_device)
    gray_model = gray_model.to(torch_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gray_model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        if epoch % DECAY_RATE == 0:
            learning_rate *= LR_DECAY
            optimizer = optim.Adam(gray_model.parameters(), lr=learning_rate)
            print(f'New learning rate: {learning_rate}')
        train_loss = train_epoch_gray(gray_model, color_model, train_loader, criterion, optimizer, torch_device)
        val_loss = val_epoch_gray(gray_model, color_model, val_loader, criterion, torch_device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch: {epoch}\tTrainLoss: {train_loss}\tValLoss: {val_loss}')
        if epoch % CHECKPOINT_RATE == 0:
            print('Checkpointing model...')
            checkpoint(gray_model, os.path.join(MODEL_DIR, f'gray_{epoch}.pt'))
