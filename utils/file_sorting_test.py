import os
import shutil
import pandas as pd

from utils.config import *
from utils.config_test import *

##MRI########
test_dir = test_cfg.test_dir_MRI
test_class_names0 = os.listdir(test_dir)
test_class_names = sorted(test_class_names0)
print(test_class_names)
test_num_class = len(test_class_names)
test_image_files = [[os.path.join(test_dir, test_class_name, x) 
               for x in os.listdir(os.path.join(test_dir, test_class_name))] 
               for test_class_name in test_class_names]

M_test_image_file_list = []
M_test_image_label_list = []
for i, class_name in enumerate(test_class_names):
    M_test_image_file_list.extend(test_image_files[i])
    M_test_image_label_list.extend([i]*len(test_image_files[i]))
values_to_count = [0, 1]
counts = [M_test_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for testing in MRI: {count}")  
    
#Xray######
test_dir = test_cfg.test_dir_Xray
test_class_names0 = os.listdir(test_dir)
test_class_names = sorted(test_class_names0)
print(test_class_names)
test_num_class = len(test_class_names)
test_image_files = [[os.path.join(test_dir, test_class_name, x) 
               for x in os.listdir(os.path.join(test_dir, test_class_name))] 
               for test_class_name in test_class_names]

X_test_image_file_list = []
X_test_image_label_list = []
for i, class_name in enumerate(test_class_names):
    X_test_image_file_list.extend(test_image_files[i])
    X_test_image_label_list.extend([i]*len(test_image_files[i]))
values_to_count = [0, 1]
counts = [ X_test_image_label_list.count(value) for value in values_to_count]
for value, count in zip(values_to_count, counts):
    print(f"Count of {value} for testing in X-ray: {count}")  
    
#Sorting
M_test_image_file_list = sorted(M_test_image_file_list)
M_test_image_label_list = sorted(M_test_image_label_list)
X_test_image_file_list = sorted(X_test_image_file_list)
X_test_image_label_list = sorted(X_test_image_label_list)

#Save the file
k = pd.DataFrame(dict({"MRI_volume_name":M_test_image_file_list, "Xray_image_name": X_test_image_file_list}))
k.to_csv(test_cfg.folder_path+"test_list.csv")

