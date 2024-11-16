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
from utils.config import *

class Test_CONFIG():
  seg_dir = "./seg/model.pth"
  MRI_dir_path = "./MRI_volumes_test/" 
  Xray_dir_path = "./Xray_images_test/"
  save_path = "./results/"
  test_dir_MRI= MRI_dir_path + 'Test' 
  test_dir_Xray = Xray_dir_path + 'test'
  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_name ="results1" 
  folder_path = save_path + model_name
  embeddings_path_png = folder_path + "/" + model_name + "embeddings.png"
  embeddings_path = folder_path + "/" + model_name + "embeddings.npy"
  embeddings_path_png = folder_path + "/" + model_name + "tSNE.png"
  pretrained = True

  num_classes = 2

  batch_size_test = 1

  max_epochs = 1

  num_workers = 1

test_cfg = Test_CONFIG()


