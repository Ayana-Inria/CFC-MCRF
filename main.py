# -*- coding: utf-8 -*-
"""
Main script: build the dataset, train the selected network, and extract tensors.
"""

import os
import torch
import torch.utils.data as data

from dataset import PRISMA_dataset
from net.net import FCN_lambda, FCN_lambda1, UNet_lambda
from net.crossmodal_net import CMFNet
from train import train
from extract import extract_tensors
import train as train_module


############# Parameters to edit ###############

input_folder = 'data'
output_folder = 'checkpoints/'
prisma_tensors_folder = 'PRISMA_Tensors'

train_ids = [
    # 'tile_name_without_suffix',
]

extract_ids = train_ids

architecture = 'fcn'       # 'fcn', 'fcn1', 'unet', or 'crossmodal'
use_ms_loss = False        # True to use train_MS
n_channels = 1             # PAN channels
n_hsi_channels = 234       # HSI channels after band selection
n_classes = 7
batch_size = 16
epochs = 100
learning_rate = 1e-4
save_epoch = 10
cache = False
augmentation = True


############# Main ###############

os.makedirs(output_folder, exist_ok=True)
os.makedirs(prisma_tensors_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = PRISMA_dataset(
    train_ids,
    folder=input_folder,
    cache=cache,
    augmentation=augmentation,
)

train_loader = data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

if architecture == 'fcn':
    net = FCN_lambda(n_channels, n_classes)
elif architecture == 'fcn1':
    net = FCN_lambda1(n_channels, n_classes)
elif architecture == 'unet':
    net = UNet_lambda(n_channels, n_classes)
elif architecture == 'crossmodal':
    net = CMFNet(in_channels=n_hsi_channels, out_channels=n_classes)
else:
    raise ValueError('Unknown architecture: {}'.format(architecture))

net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Keep the original train.py style: the training loop uses global variables.
train_module.train_loader = train_loader
train_module.output_folder = output_folder

if use_ms_loss:
    train_MS(net, optimizer, epochs, save_epoch=save_epoch)
else:
    train(net, optimizer, epochs, save_epoch=save_epoch)

checkpoint_path = os.path.join(output_folder, architecture + '_final.pth')
torch.save(net.state_dict(), checkpoint_path)
print('Saved {}'.format(checkpoint_path))

extract_tensors(
    net,
    extract_ids,
    folder=input_folder,
    save_folder=prisma_tensors_folder,
)
