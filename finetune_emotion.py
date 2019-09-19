#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, ToTensor
from model import init_grayscale_resnet, train_epoch, val_epoch, checkpoint

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Main script
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    NUM_EMOTIONS = 5
    P_TRAIN = 0.7
    BATCH_SIZE = 360
    NUM_WORKERS = 10
    LEARNING_RATE = 0.00005
    NUM_EPOCHS = 100
    CHECKPOINT_RATE = 5  # number of epochs after which to checkpoint the model

    IMG_DIR = '/home/mchobanyan/data/emotion/images/gray/'
    BASE_MODEL_PATH = '/home/mchobanyan/data/emotion/models/emotion_detect/fer-finetune-all/model_10.pt'
    MODEL_DIR = '/home/mchobanyan/data/emotion/models/emotion_detect/gray-base/'

    model = init_grayscale_resnet()
    conv_out_features = model.fc.in_features
    model.fc = nn.Linear(conv_out_features, NUM_EMOTIONS)
    model.load_state_dict(torch.load(BASE_MODEL_PATH))

    layers = list(model.children())
    for i in range(len(layers) - 3):
        for param in layers[i].parameters():
            param.requires_grad = False

    transforms = Compose([RandomHorizontalFlip(),
                          RandomAffine(degrees=10, translate=(0.25, 0.25), scale=(0.5, 1)),
                          ToTensor()])
    dataset = ImageFolder(IMG_DIR, transform=transforms, loader=Image.open)
    labels = dataset.classes

    train_size = int(len(dataset) * P_TRAIN)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    device = torch.device('cuda')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        message = f'Epoch: {epoch}\tTrainLoss: {train_loss}'
        if len(val_data) > 0:  # only run validation epoch if the validation dataset is not empty
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            message += f'\tValLoss: {val_loss}\tValAcc: {val_acc}'
        print(message)
        if epoch % CHECKPOINT_RATE == 0:
            print('Checkpointing model...')
            checkpoint(model, os.path.join(MODEL_DIR, f'model_{epoch}.pt'))
