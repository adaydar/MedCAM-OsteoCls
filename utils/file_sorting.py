import os
import shutil
import pandas as pd

from utils.config import *

#import train
#MRI###################################################################################################
class_names0 = os.listdir(cfg.train_dir_M)
class_names = sorted(class_names0)
print(class_names)
num_class = len(class_names)
image_files = [[os.path.join(cfg.train_dir_M, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir_M, class_name))] 
               for class_name in class_names]
print(num_class)

MRI_image_file_list = []
MRI_image_label_list = []
for i, class_name in enumerate(class_names):
    MRI_image_file_list.extend(image_files[i])
    MRI_image_label_list.extend([i] * len(image_files[i]))
values_to_count = [0, 1]
counts = [MRI_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for training in MRI: {count}")

#Xray##################################################################################################
class_names0 = os.listdir(cfg.train_dir_X)
class_names = sorted(class_names0)
print(class_names)
num_class = len(class_names)
image_files = [[os.path.join(cfg.train_dir_X, class_name, x) 
               for x in os.listdir(os.path.join(cfg.train_dir_X, class_name))] 
               for class_name in class_names]
#print(num_class)

X_image_file_list = []
X_image_label_list = []
for i, class_name in enumerate(class_names):
    X_image_file_list.extend(image_files[i])
    X_image_label_list.extend([i] * len(image_files[i]))
values_to_count = [0, 1]
counts = [X_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for training in X-ray: {count}")

#MRI##################################################################################################### 
#import valid
v_class_names0 = os.listdir(cfg.val_dir_M)
v_class_names = sorted(v_class_names0)
#print(v_class_names)
v_num_class = len(v_class_names)
v_image_files = [[os.path.join(cfg.val_dir_M, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir_M, v_class_name))] 
               for v_class_name in v_class_names]

MRI_v_image_file_list = []
MRI_v_image_label_list = []
for i, class_name in enumerate(v_class_names):
    MRI_v_image_file_list.extend(v_image_files[i])
    MRI_v_image_label_list.extend([i]*len(v_image_files[i]))
values_to_count = [0, 1]
counts = [MRI_v_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for valid in MRI: {count}")    

#Xray####################################################################################################
v_class_names0 = os.listdir(cfg.val_dir_X)
v_class_names = sorted(v_class_names0)
#print(v_class_names)
v_num_class = len(v_class_names)
v_image_files = [[os.path.join(cfg.val_dir_X, v_class_name, x) 
               for x in os.listdir(os.path.join(cfg.val_dir_X, v_class_name))] 
               for v_class_name in v_class_names]

X_v_image_file_list = []
X_v_image_label_list = []
for i, class_name in enumerate(v_class_names):
    X_v_image_file_list.extend(v_image_files[i])
    X_v_image_label_list.extend([i]*len(v_image_files[i]))
values_to_count = [0, 1]
counts = [X_v_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for valid in X-ray: {count}")  
    
#File sorting
##MRI#########################
MRI_image_file_list = sorted(MRI_image_file_list)
MRI_image_label_list = sorted(MRI_image_label_list)
MRI_v_image_file_list = sorted(MRI_v_image_file_list)
MRI_v_image_label_list = sorted(MRI_v_image_label_list)
###Xray################
X_image_file_list = sorted(X_image_file_list) 
X_image_label_list = sorted(X_image_label_list)
X_v_image_file_list = sorted(X_v_image_file_list)
X_v_image_label_list = sorted(X_v_image_label_list)   
   
#Save the file
k = pd.DataFrame(dict({"MRI_volume_name":MRI_image_file_list, "Xray_image_name": X_image_file_list}))
k.to_csv(cfg.folder_path+"train_list.csv")
k = pd.DataFrame(dict({"MRI_volume_name":MRI_v_image_file_list, "Xray_image_name": X_v_image_file_list}))
k.to_csv(cfg.folder_path+"val_list.csv")

