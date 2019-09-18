#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor
from dtsckit.pytorch.model import checkpoint
from model import init_grayscale_resnet

# imagenet color normalization
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDVS = [0.229, 0.224, 0.225]


class FakeImagenetDataset(ImageFolder):
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
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    dataset = FakeImagenetDataset(image_dir='/home/mchobanyan/data/emotion/images/imagenet/',
                                  colored_transform=Compose([ToTensor(), Normalize(IMAGENET_MEANS, IMAGENET_STDVS)]),
                                  gray_transform=ToTensor())
    print(f'Number of images: {len(dataset)}')

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    bsize = 32
    train_loader = DataLoader(train_data, batch_size=bsize, num_workers=6, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bsize, num_workers=6)

    color_model = resnet50(pretrained=True)
    gray_model = init_grayscale_resnet()

    # only allow the first convolution layer in the gray model to train
    for param in gray_model.parameters():
        param.requires_grad = False
    gray_model.conv1.weight.requires_grad = True

    device = torch.device('cuda')
    color_model = color_model.to(device)
    gray_model = gray_model.to(device)
    criterion = nn.MSELoss()

    learning_rate = 0.001
    optimizer = optim.Adam(gray_model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    num_epochs = 50
    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            learning_rate /= 2
            optimizer = optim.Adam(gray_model.parameters(), lr=learning_rate)
            print(f'New learning rate: {learning_rate}')

        train_loss = train_epoch_gray(gray_model, color_model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = val_epoch_gray(gray_model, color_model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch}\tTrainLoss: {train_loss}\tValLoss: {val_loss}')

        if epoch % 5 == 0:
            print('Checkpointing model...')
            checkpoint(gray_model, f'/home/mchobanyan/data/emotion/models/emotion_detect/imagenet/gray_{epoch}.pt')
