# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 17:34:32 2022

@author: marti
"""

import torch
import torch.nn as nn 
from utils.utils_network import *

class FCN_λ(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FCN_λ, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # PAN channel processing
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down_ter(32, 64)
        self.drop = nn.Dropout(0.5)
        
        # HYS channel processing
        self.deep1 = Conv_lambda(1, 1)  # spectral max pooling 3D, (3, 1, 1)
        self.deep2 = Conv_lambda(1, 1)  # 26 channels HYS
        
        # Bottleneck -> data fusion
        self.bottle = DoubleConv(88, 64)

        
        # decoder PAN + HYS
        factor = 2 if bilinear else 1
        #self.up1 = Up_ter(120, 64 // factor, bilinear)
        self.up1 = Up_ter(96, 64 // factor, bilinear)
        self.up2 = Up(48, 32 // factor, bilinear)
        self.outc = OutConv(16, n_classes)


    def forward(self, x, y):

        activations = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y1 = self.deep1(torch.unsqueeze(y, dim=1))
        y2 = self.deep2(y1)
        x4 = self.bottle(torch.cat((x3, torch.squeeze(y2, dim=1)), 1))
        x4 = self.drop(x4)
        x = self.up1(x4, x2)
        activations.append(x)
        x = self.up2(x, x1)
        activations.append(x)
        logits = self.outc(x)
        return logits, activations
    
    

class FCN_λ1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FCN_λ1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # PAN channel processing
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down_ter(32, 64)
        self.drop = nn.Dropout(0.5)
        
        # HYS channel processing
        self.deep1 = Conv_lambda(1, 1)  # spectral max pooling 3D, (3, 1, 1)
        
        # Bottleneck -> data fusion
        self.bottle = DoubleConv(140, 96)

        
        # decoder PAN + HYS
        factor = 2 if bilinear else 1
        self.up1 = Up_ter(128, 64 // factor, bilinear)
        self.up2 = Up(48, 32 // factor, bilinear)
        self.outc = OutConv(16, n_classes)


    def forward(self, x, y):

        activations = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        y1 = self.deep1(torch.unsqueeze(y, dim=1))
        x4 = self.bottle(torch.cat((x3, torch.squeeze(y1, dim=1)), 1))
        x4 = self.drop(x4)
        x = self.up1(x4, x2)
        activations.append(x)
        x = self.up2(x, x1)
        activations.append(x)
        logits = self.outc(x)
        return logits, activations
   
    
   
    
class UNet_λ(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_λ, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down_ter(128, 256) # original 128 + 50 HSI bands
        self.down3 = Down(332, 512) # original 256 + 78 HSI bands
        self.deep1 = Conv_lambda(1, 1)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up_ter(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, y):

        activations = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        y1 = self.deep1(torch.unsqueeze(y, dim=1))
        x4 = self.down3(torch.cat((torch.squeeze(y1, dim=1), x3), 1))
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        activations.append(x)
        x = self.up2(x, x3)
        activations.append(x)
        x = self.up3(x, x2)
        activations.append(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, activations

# ASCII aliases for easier imports from main.py
FCN_lambda = FCN_λ
FCN_lambda1 = FCN_λ1
UNet_lambda = UNet_λ
