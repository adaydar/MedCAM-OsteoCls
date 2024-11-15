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
import torchvision.transforms as T

from utils.config import *
from utils.config_test import *
from utils.file_sorting import *
from utils.file_sorting_test import *
from utils.Segmentation_model import *

#Dataloaders
custom_mapping = {
    'R': 0, 'L':1 
}

# Function to convert each character
def convert_char(char):
    if char.isdigit():
        return int(char)
    elif char in custom_mapping:
        return custom_mapping[char]
    else:
        raise ValueError(f"Unsupported character: {char}")

def custom_collate(batch):
    # Extract the first output (volumes) from each sample
    volumes = [item[0] for item in batch]
    # Find the maximum number of slices in the batch
    #max_slices = max([volume.shape[1] for volume in volumes])
    max_slices=35
    # Pad volumes to have the same number of slices
    padded_volumes = []
    for volume in volumes:
        if volume.shape[1] < max_slices:
            # Calculate the padding size
            padding = (0, 0, 0, max_slices - volume.shape[1], 0, 0)
            # Pad the volume with zeros (or any other value if specified)
            padded_volume = F.pad(volume, padding, "constant", 0)
            #print(padded_volume.shape)
            padded_volumes.append(padded_volume)
        else:
            padded_volumes.append(volume)
    
    xray_volumes = [item[1] for item in batch]
    padded_volumes_xray = []
    for v in xray_volumes:
         padded_volumes_xray.append(v)       
    padded_volumes_xray = torch.stack(padded_volumes_xray)
    
    labels_xray = [item[2] for item in batch] 
    padded_labels_xray = []
    for label in labels_xray:
        label = torch.tensor(label)
        padded_labels_xray.append(label)
    padded_labels_xray = torch.stack(padded_labels_xray)  
      
    names_xray = [item[3] for item in batch] 
    padded_name_xray = []
    for name in names_xray:
        #name = [ord(char) for char in name]
        #name = [convert_char(char) for char in name]
        name = name.replace('R', '0').replace('L', '1')
        name1 = torch.tensor([int(name)])
        padded_name_xray.append(name1)
    names_xray = torch.stack(padded_name_xray)  
    # Stack the padded volumes into a single tensor
    padded_volumes = torch.stack(padded_volumes)
    # Return a tuple of the padded volumes and other outputs
    return padded_volumes, padded_volumes_xray, padded_labels_xray, names_xray
    
class Combined_Dataset(Dataset):
     def __init__(self, MRI_image_file_list, MRI_label_file_list, xray_image_file_list, xray_label_file_list, MRI_transforms, xray_transforms):
         ##MRI######
         #self.subdirectories = sorted(os.listdir(seg_dir))
         self.MRI_image_file_list = MRI_image_file_list
         self.MRI_label_file_list = MRI_label_file_list
         self.MRI_transforms = MRI_transforms
         self.MRI_pd_data =pd.DataFrame(self.MRI_label_file_list)
         #Xray##########
         self.xray_image_file_list = xray_image_file_list
         self.xray_label_file_list = xray_label_file_list
         self.xray_transforms = xray_transforms
         self.pd_data =pd.DataFrame(self.xray_label_file_list)
         #Seg model
         self.seg_model = seg_model 
       
     def __len__(self):
         return len(self.MRI_label_file_list)
     def __getitem__(self,index):
         MRI_image = np.load(self.MRI_image_file_list[index])
         image_name = self.xray_image_file_list[index]
         image_n = os.path.basename(image_name).split('.')[0]
         xray_image = Image.open(self.xray_image_file_list[index]).convert('RGB')
         xray_label = self.xray_label_file_list[index]
         #print(type(xray_label))
         return self.MRI_transforms(MRI_image), self.xray_transforms(xray_image), xray_label, image_n


class RotateVolume90:
    def __call__(self, volume):
     rotated_volume = np.zeros_like(volume)
     for i in range(volume.shape[0]):
            image = Image.fromarray(volume[i,:,:])
            rotated_image = S.rotate(image, 90)
            rotated_image = S.vflip(rotated_image)
            rotated_volume[i, :, :] = np.array(rotated_image)
     return rotated_volume
    
###MRI_transformation######################
MRI_transformation = T.Compose([ T.Lambda(lambda x: RotateVolume90()(x)) , T.ToTensor()]) #T.Lambda(rotate_volume_90)
###Xray_transformation ####################
pixel_mean, pixel_std = 0.66133188, 0.21229856

Xray_test_transforms= T.Compose([
            T.Resize((cfg.Image_dim,cfg.Image_dim)),
            T.ToTensor(),
            T.Normalize([pixel_mean]*3, [pixel_std]*3)])
                   
test_dataset = Combined_Dataset(M_test_image_file_list, M_test_image_label_list, X_test_image_file_list, X_test_image_label_list, MRI_transforms=MRI_transformation,xray_transforms = Xray_test_transforms)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_cfg.batch_size_test, shuffle=False, num_workers=test_cfg.num_workers, drop_last=False, collate_fn=custom_collate)
