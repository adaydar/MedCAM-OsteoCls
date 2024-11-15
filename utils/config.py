import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import random
import torchvision.transforms as T
import torch.nn.functional as F

cv2.setRNGSeed(0)
print(cv2.__version__)

to_pil = T.ToPILImage()

# Fix all random seeds.
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(1)

class CONFIG():
  seg_dir = "./seg/model.pth" #import segmentation model's .pth file format to "seg" folder in main directory. The segmentation model can be obtained from 
  """https://github.com/adaydar/MtRA-Unet/tree/main"""
    
  MRI_dir_path = "./MRI_volumes/" #Keep sagittal MRI slices in .npy format to "MRI_volumes" folder in main directory.  
  Xray_dir_path = "./Xray_images/" #Keep sagittal X-ray images in .png files to "Xray_images" folder in main directory.
  save_path = "./results/" #make "results" folder to save all the results. 
  GCNN_dir = "./cartilage_tear/cartilage_tearGCNN1.pth" # Keep pretrained "GCNT" model in "cartilage_tear" folder in main directory. 
  train_dir_M = MRI_dir_path + 'Train'
  train_dir_X =  Xray_dir_path + 'train'
  val_dir_M =  MRI_dir_path + 'Val'
  val_dir_X = Xray_dir_path + 'val'

  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name = "MedCAM_OsteoCls" #create "MedCAM-Osteocls" folder inside "results" folder in main directory.
  folder_path = save_path + model_name
  save_folder = folder_path+ "/" + model_name  + ".pth"
  save_folder1 = folder_path+ "/" + model_name + "_loss.png"
  save_folder2 = folder_path+ "/" + model_name + "_cm.png"
  save_folder3 =folder_path + "/" + model_name + "_class_report.csv"
  save_folder4 =folder_path + "/" + model_name + "_Gradcam.png"
  save_folder5 =folder_path + "/" + model_name + "summary.txt"
  save_folder6 = folder_path+ "/" + model_name  + ".pth"
  save_folder7 = folder_path+ "/" + model_name  + "_cs.csv"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  patch_path = "./patch_tensors1/" #Keep all the generated patch tensors here.
  log_root_folder = folder_path + "/log"
  pretrained = True

  num_classes = 2

  batch_size_train = 36
  batch_size_val = 36
  Image_dim = 300 
  max_epochs = 50
  lr = 5e-4
  weight_decay = 5e-3
  lr_decay_epoch = 5 
  momentum = 0.9

  num_workers = 1
  patience = 5
  log_every = 100
  lr_scheduler = "plateau"
  medio_lateral_regularization = 1.5
  erosion_dilation_iteration = 5
  num_patches = 10
  PATCH_SIZE = 50
  num_contour_points = 100
  
cfg = CONFIG()

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs) 
        inputs_1= inputs.permute(1,0,2,3)
        targets_1 = targets.permute(1,0,2,3)
        #print(inputs_1.shape)
        BAL1,BAL2,BAL3,BAL4 = BAWLoss(inputs_1,targets_1) 
        BAL = (BAL1*0.1+BAL2*0.2+BAL3*0.3+BAL4*0.4)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs + targets).sum() + 1e-8  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        #BAL1,BAL2,BAL3,BAL4 = BAWLoss(inputs,targets)
        Dice_BCE = 0.7*(BCE + dice_loss)+0.3*(BAL)
        
        return Dice_BCE

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 16)
        self.conv3 = conv_block(16, 16)
        self.conv4 = conv_block(16, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x4, x3, x2, x1
        
class CycleNet(nn.Module):
    def __init__(self):
        super(CycleNet, self).__init__()
        self.b_block = block()
    def forward(self, x):
        t1,t2,t3,t4 = self.b_block(x)
        return t1,t2,t3,t4  

       
def BAWLoss(preds,GT):
       Loss_1 = 0.0
       Loss_2 = 0.0
       Loss_3 = 0.0
       Loss_4 = 0.0
       #preds = preds.to(DEVICE)
       #GT = torch.squeeze(GT)
       LossL1 = nn.L1Loss()
       preds = preds.permute(1,0,2,3)
       GT = GT.permute(1,0,2,3)
       loss_net = CycleNet().to(DEVICE)
       t1,t2,t3,t4 = loss_net(preds)
       tg1,tg2,tg3,tg4 = loss_net(GT)
       Loss_1=LossL1(tg1,t1)
       Loss_2=LossL1(tg2,t2)
       Loss_3=LossL1(tg3,t3)
       Loss_4=LossL1(tg4,t4)       
       return  Loss_1,Loss_2,Loss_3,Loss_4

def dice_calc(gt,pred) :
    pred = F.sigmoid(pred)
    pred = ((pred) >= .5).float()
    #gt = ((gt) >= .5).float()
    dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
    VD = 100*(pred.sum()- gt.sum())/(gt.sum()+1e-8)
    intersection = (pred * gt).sum()
    total = (pred + gt).sum()
    union = total - intersection 
    iou = (intersection + 1e-8)/(union + 1e-8)
    VOE = 1- ((2 * (pred * gt).sum()) / ((pred+ gt).sum() + 1e-8)/(2-((2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8))))
    
    return dice_score,VD,VOE
