import os
from PIL import Image
import numpy as np
import cv2
import torch
#!pip install monai
from monai.transforms import *
from monai.data import Dataset, DataLoader
import pandas as pd
from torchvision import datasets,transforms,models
import torchvision.transforms as datasets
import torch.nn.functional as F
from torchvision.transforms import functional as S
import torch.nn as nn

from utils.config import *
from utils.CG_SSP import *

######## Classification model ########################################################################        

group_norm_layer = nn.GroupNorm(14,14).to(cfg.device)

torch.autograd.set_detect_anomaly(True)

model1 = models.vgg19(pretrained=cfg.pretrained).to(cfg.device)

model1.features[0] = torch.nn.Conv2d(14, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(cfg.device)

model2 = models.vgg19(pretrained=cfg.pretrained).to(cfg.device)
  
class MedCAM_OsteoCls(nn.Module):   
    def __init__(self, model1, model2, mrselector, in_channels, out_channels, cross_attn, num_classes=cfg.num_classes):
        super(MedCAM_OsteoCls,self).__init__()
        self.vgg19_mr = model1
        self.vgg19_x = model2
        self.mrselector = mrselector
        self.cross_attn = cross_attn
        self.avg_pool = nn.AdaptiveAvgPool2d((9, 9))
        #self.max_pool = nn.Conv2d(in_channels=14336, out_channels=14336//2,kernel_size=1,stride=1,padding=1)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        encoder_norm = nn.LayerNorm(512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.row_embed = nn.Parameter(torch.rand(50, 512))
        self.col_embed = nn.Parameter(torch.rand(50, 512))
        self.weight1 = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))
        #self.adj_conv = nn.Conv2d(1024,512,kernel_size=1)
             
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024//2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024//2, num_classes),
        ).to(cfg.device)      

    def forward(self, x, x1,im_name):
         x = x.permute(0,2,1,3) 
         x = self.mrselector(x,x1, im_name)
         x = self.vgg19_mr.features(x) 
         x1 = self.vgg19_x.features(x1)  
         x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1)) # Shape: (batch_size, hidden_dim, 1, 1) 
         x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) 
         h, w = x.shape[-2:]
         pos = torch.cat([
                         self.col_embed[:w].unsqueeze(0).repeat(h,1,1),
                         self.row_embed[:h].unsqueeze(0).repeat(1,w,1),
                         ], dim=1).flatten(0,1).unsqueeze(1)
                         
         b = self.transformer_encoder(pos+x.flatten(2).permute(2,0,1))
         b = b.permute(1,0,2)
         
         h, w = x1.shape[-2:]
         pos = torch.cat([
                         self.col_embed[:w].unsqueeze(0).repeat(h,1,1),
                         self.row_embed[:h].unsqueeze(0).repeat(1,w,1),
                         ], dim=1).flatten(0,1).unsqueeze(1)
                         
         b1 = self.transformer_encoder(pos+x1.flatten(2).permute(2,0,1))
         b1 = b1.permute(1,0,2)
         b1 = torch.flatten(b1,1)
         b = torch.flatten(b, 1)
         z = self.cross_attn(b,b1)
         x5 = self.classifier(z)
         return x5,z
                 
num_classes = 2  # Adjust this according to your problem
model = MedCAM_OsteoCls(model1, model2, mrselector, 14, 42, cross_attn,num_classes=cfg.num_classes).to(cfg.device)


