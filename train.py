# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 18:52:11 2022

@author: marti
"""

from IPython.display import clear_output
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.init
from torch.autograd import Variable
from net.net import *
from utils.utils import *
from net.loss import *
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_dataset import *
try:
    from net.test_network import test
except ImportError:
    test = None
import cv2

 
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 10):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    device = next(net.parameters()).device
    if weights is not None:
        weights = weights.to(device)

    criterion = nn.NLLLoss(weight=weights)
    iter_ = 0
    
    for e in tqdm(range(1, epochs + 1)):

        net.train()
        for batch_idx, (data_pan, data_hsi, target) in enumerate(train_loader):
            data_pan, data_hsi, target = Variable(data_pan.to(device)), Variable(data_hsi.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            output = net(data_pan, data_hsi)[0]
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()  #loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            iter_ += 1
            
            del(data_pan, data_hsi, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            torch.save(net.state_dict(), output_folder + 'test_epoch{}'.format(e))
    torch.save(net.state_dict(), output_folder + 'test_final')