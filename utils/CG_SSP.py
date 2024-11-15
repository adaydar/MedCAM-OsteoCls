from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as S
import torch.nn as nn
import torch_geometric
import torchvision.transforms as T
from scipy.spatial import distance
import sys

from utils.Segmentation_model import *
from utils.GCNT import *
from utils.XMRCA import *

import cv2
from cv2 import findContours 
from cv2 import RETR_EXTERNAL
from cv2 import CHAIN_APPROX_SIMPLE
from cv2 import THRESH_BINARY
from cv2 import threshold
from cv2 import contourArea
from cv2 import FILLED
from cv2 import drawContours
from cv2 import line 
from cv2 import dilate
from cv2 import erode
from cv2 import arcLength
from cv2 import copyMakeBorder
from cv2 import BORDER_CONSTANT
from cv2 import resize as res
from cv2 import imwrite 
from cv2 import resize
from cv2 import GaussianBlur
from cv2 import equalizeHist
from cv2 import Canny
from cv2 import COLOR_RGB2GRAY
from cv2 import cvtColor
from cv2 import LUT


#Erosion_and_dilation

def Erosion_and_dilation(binary,contours,Kernel_Size):
  largest_contour = max(contours, key=contourArea)
  if len(contours) <= 1:
   eroded = binary 
   pass
  else:
   # Step 6: Identify the smallest contour
   min_area = float('inf')
   min_contour = None
   for contour in contours:
     area = contourArea(contour)
     if area < min_area:
         min_area = area
         min_contour = contour
   if min_contour is not None:
     drawContours(binary, [min_contour], -1, (0), thickness=FILLED)
   contours = sorted(contours, key=contourArea, reverse=True)
   if len(contours) > 2:
     contours = contours[:2]   
   contour_image = np.zeros_like(binary)    
   drawContours(contour_image, [min_contour], -1, (255, 255, 255), thickness=2)
   if len(contours) > 1:
     contour1 = contours[0]
     contour2 = contours[1] 
     min_distance = float('inf')
     point1 = None
     point2 = None
     for pt1 in contour1:
         for pt2 in contour2:
             dist = distance.euclidean(pt1[0], pt2[0])
             if dist < min_distance:
                 min_distance = dist
                 point1 = pt1[0]
                 point2 = pt2[0]    
     kernel = np.ones(Kernel_Size, np.uint8)
     dilated = dilate(binary, kernel, iterations=cfg.erosion_dilation_iteration)
     eroded = erode(dilated, kernel, iterations=cfg.erosion_dilation_iteration)
   else:
     eroded = binary
  return eroded, largest_contour
  
def calculate_area(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_patch_coordinates_from_max_arc(slice_imgs):
    max_arc_length = 0
    max_arc_slice = None
    
    for slice_img in slice_imgs:
        # Convert to numpy and find contours
        slice_np = slice_img.numpy().astype(np.uint8)
        _, binary = cv2.threshold(slice_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            arc_length = cv2.arcLength(contour, True)
            if arc_length > max_arc_length:
                max_arc_length = arc_length
                max_arc_slice = slice_img

def pad_image(image, target_size=(50, 50)):
    target_h, target_w = target_size
    h, w = image.shape
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    # Calculate padding: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    
    # Pad the image
    padded_image = F.pad(image, padding)
    return padded_image
                         
def centroid_gen(image,mask,largest_contour,contour_points, PATCH_SIZE):    
    num_points = cfg.num_contour_points
    contour_length = arcLength(largest_contour, True)
    interval = contour_length / num_points
    sampled_points = []
    for i in range(num_points):
       point = contour_points[i * len(contour_points) // num_points]
       sampled_points.append(point)
    centroids = []
    for i in range(len(sampled_points) // 2):
       point1 = sampled_points[i]
       point2 = sampled_points[-i-1]
       centroid = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
       centroids.append(centroid)
    for i in range(len(centroids) - 1):#
       line(mask, centroids[i], centroids[i + 1], (0, 255, 0), 2)#       
    patches = []
    selected_centroids = np.linspace(0, len(centroids) - 1, 10, dtype=int) 
    patch_size = PATCH_SIZE #50
    patch_coordinates=[]
    for idx, center_idx in enumerate(selected_centroids):
       center = centroids[center_idx]
       top_left = (center[0] - patch_size // 2, center[1] - patch_size // 2)
       bottom_right = (center[0] + patch_size // 2, center[1] + patch_size // 2)
       patch = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
       patch_top = max(0, top_left[1])
       patch_bottom = min(image.shape[0], bottom_right[1])
       patch_left = max(0, top_left[0])
       patch_right = min(image.shape[1], bottom_right[0])
       patch_coordinates.append((patch_top,patch_left,patch_bottom,patch_right))
    return patch_coordinates


# Function to calculate arc length and get patch coordinates
def get_patch_coordinates_from_max_arc(slice_imgs,seg):
    max_arc_length = 0
    max_arc_slice = None
    mask_contour_dict = {}  
    slice_img_list = []
    mask_2_list = []
    largest_c_list = []
    largest_c_array_list = []
    num_slices = slice_imgs.shape[0]
    for slice_index in range(num_slices):
        slice_img = slice_imgs[slice_index]
        slice_img = slice_img.unsqueeze(0).unsqueeze(0)
        slice_img_list.append(slice_img)
        mask,_,_,_,_ = seg(slice_img)  # Process with the segmentation model 
        mask = torch.sigmoid(mask)          
        mask_2 = (mask * 255).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        contours, _ = findContours(mask_2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        #print(type(contours))
        if len(contours) ==0:
         largest_c = np.zeros((0, 1, 2), dtype=np.int32)
         #print(largest_c)
        else:
         mask_2, largest_c = Erosion_and_dilation(mask_2,contours,5)  #(problems need to be solved)         
        contour_length = arcLength(largest_c, closed=True) if largest_c.shape[0] != 0 else 0 #arcLength(largest_c, True)  

        mask_2_list.append(mask_2)
        largest_c_list.append(contour_length)
        largest_c_array_list.append(largest_c)
    mask_contour_dict = {img: (mask, contours,contours_array) for img, mask, contours, contours_array in zip(slice_img_list, mask_2_list, largest_c_list,largest_c_array_list)}  
    # Extract the contour counts
    contour_counts = [contours for _,contours,_ in mask_contour_dict.values()]
    #print(contour_counts)
    # Find the maximum contour count
    max_contour = max(contour_counts)   
    max_coutour_array = next(contours_array for slice_img, (mask, contour,contours_array) in mask_contour_dict.items() if contour == max_contour)
    max_slice = next(slice_img for slice_img, (mask, contour,contours_array) in mask_contour_dict.items() if contour == max_contour)
    max_slice_np = (max_slice*255).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    #imwrite(cfg.folder_path + "/max_image/" + "max.png", max_slice_np) 
    
    max_mask = next(mask for slice_img, (mask, contour,contours_array) in mask_contour_dict.items() if contour == max_contour)   
    #imwrite(cfg.folder_path + "/max_masks/" + "max.png", max_mask) 
    
    contour_points = max_coutour_array[:, 0, :]
    return max_slice_np, max_mask, max_coutour_array, contour_points,max_slice
                        

def patch_gen(patch_coordinates,image_tensor, num_patches, each_batch,im_name1):
 all_cropped_patches = []
 # Iterate over each image in the batch (batch size is 1 here, but generalize for more)
 for img_idx in range(image_tensor.size(1)):  # Loop over each image/slice
    image_patches = []
    
    # Loop over each set of cropping coordinates
    for patch_idx, coord in enumerate(patch_coordinates[:num_patches]):  # Limit to num_patches
        y1, x1, y2, x2 = coord

        cropped_patch = image_tensor[0, img_idx, y1:y2, x1:x2]
        #print(cropped_patch.shape)
        cropped_patch = pad_image(cropped_patch)
        cropped_patch_np = (cropped_patch*255).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        #imwrite(cfg.folder_path + "/patches/" + str(im_name1.item()) + "_" +str(img_idx) + "_" + str(patch_idx) + "_patches.png", cropped_patch_np) 
        
        # Append the cropped patch to the list
        image_patches.append(cropped_patch)
    
    # Stack all patches for this image
    image_patches_tensor = torch.stack(image_patches)
    
    # Append this image's patches to the list of all patches
    all_cropped_patches.append(image_patches_tensor)

 # Convert the list of all image patches to a single tensor
 all_cropped_patches_tensor = torch.stack(all_cropped_patches)
 #print(all_cropped_patches_tensor.shape)
 torch.save(all_cropped_patches_tensor, cfg.patch_path + str(im_name1.item()) + ".pt")
 return all_cropped_patches_tensor

def GCNN_for_patches(all_cropped_patches_tensor, each_batch): 
 append_scores_list = []
 GCNN_outputs_list = []
 for ims_idx in range(all_cropped_patches_tensor.shape[0]):
    input_GCNN = all_cropped_patches_tensor[ims_idx] 
    GCNN_output = GCNN_mask(input_GCNN)
    GCNN_outputs_list.append(GCNN_output)
    os = attention_module(GCNN_output)
    #print(os)
    append_scores_list.append(os)
 GCNN_scores = torch.tensor(append_scores_list, dtype=torch.float32)
 mean_GCNN_scores = GCNN_scores.mean()
 std_GCNN_scores = GCNN_scores.std()
 normalized_score = abs(GCNN_scores - mean_GCNN_scores) / std_GCNN_scores
 scores = normalized_score.view(1,normalized_score.shape[0])
 scores[:, :7] += cfg.medio_lateral_regularization
 scores[:, -7:] += cfg.medio_lateral_regularization
 #print(scores)
 top_values, top_indices = torch.topk(scores, 14)
 sorted_indices = torch.sort(top_indices, dim=1).values
 #print(top_indices)
 #print(sorted_indices[0])
 #print(each_batch.shape)
 sorted_top_values = torch.gather(scores, 1, sorted_indices)
 #print(sorted_top_values)
 selected_slices = each_batch[sorted_indices[0],:,:]
 selected_scores = sorted_top_values
 selected_slices = selected_slices.unsqueeze(dim=0)
 #print(selected_slices.shape)
 selected_scores = selected_scores.view(1, 14, 1, 1).to(cfg.device)
 #Normalization
 mean_selected_scores = selected_scores.min()
 std_selected_scores = selected_scores.max()
 normalized_selected_scores = abs(selected_scores - mean_selected_scores) / std_selected_scores
 normalized_selected_scores = normalized_selected_scores * (0.9 - 0.1) + 0.1
 selected_slices = group_norm_layer(selected_slices)
 weighted_images =  selected_slices * normalized_selected_scores
 weighted_images = weighted_images.squeeze(dim=0)
 return weighted_images

##Filter out black images #####################
def is_black_image(image):
    # Check if all pixels are black (value 0) across all channels
    return torch.all(image == 0).item()

def filter_black_images(each_batch):
    valid_indices = []
    
    # Iterate over each image in the batch
    for i in range(each_batch.shape[0]):
        image = each_batch[i]
        if not is_black_image(image):
            valid_indices.append(i)
    
    # Select only the valid images
    filtered_batch = each_batch[valid_indices]
    
    return filtered_batch   
    
#padding the features
def adjust_feature_maps(tensor, target_n):
    current_n = tensor.shape[0]
    
    if current_n < target_n:
        # Padding
        padding = torch.zeros((target_n - current_n, 50, 50))
        padding = padding.to(cfg.device)
        adjusted_tensor = torch.cat((tensor, padding), dim=0)
    elif current_n > target_n:
        # Truncating
        adjusted_tensor = tensor[:target_n, :, :]
    else:
        # If the number of feature maps is already target_n
        adjusted_tensor = tensor
    
    return adjusted_tensor    
                                         
class CG_SSP(nn.Module):
     def __init__(self, segmentation_model,Gcnn_mask,cross_attn):
         super(CG_SSP,self).__init__()
         self.segmentation_model = segmentation_model
         self.Gcnn_mask = Gcnn_mask
         self.cross_attn = cross_attn
         
     def forward(self,x,x1,im_name):
         batch_input = []               
         #num_slices = x.shape[1]
         #print(im_name)
         num_batch = x.shape[0]
         ops=[]
         output_images=[]
         weighted=[]
         attent_feats = []
         for r in range(num_batch):           
          e_batch = x[r, :, :, :] 
          im_name1 = im_name[r]
          each_batch = filter_black_images(e_batch)
          #print(each_batch.shape[0])
          num_slices = each_batch.shape[0]
          each_batch_xray = x1[r,:,:,:]
          batch_attention_scores=[]  
          
          if not os.path.exists(cfg.patch_path + str(im_name1.item()) + ".pt"):
            #print("not_exists")   
            max_slice_np, max_mask, max_contour, contour_points, max_slice_tensor = get_patch_coordinates_from_max_arc(each_batch,self.segmentation_model) #working fine                 
            patch_coordinates = centroid_gen(max_slice_np, max_mask, max_contour,contour_points, PATCH_SIZE) 
            all_cropped_patches_tensor = patch_gen(patch_coordinates,each_batch.unsqueeze(0),cfg.num_patches, each_batch,im_name1)
          else:
            #print("yes_exists")
            all_cropped_patches_tensor = torch.load(cfg.patch_path + str(im_name1.item()) + ".pt")
            all_cropped_patches_tensor = all_cropped_patches_tensor.to(cfg.device)
          weighted_images = GCNN_for_patches(all_cropped_patches_tensor,each_batch)
          output_images.append(weighted_images)
         x = torch.stack(output_images,0)
         x = x.squeeze(1)
         return x  

mrselector = CG_SSP(seg_model,GCNN_mask,cross_attn).to(cfg.device)   
group_norm_layer = nn.GroupNorm(14,14).to(cfg.device)
