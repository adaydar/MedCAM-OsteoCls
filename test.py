import os
from PIL import Image
import numpy as np
import cv2
import torch
#!pip install monai
from monai.transforms import *
from monai.data import Dataset, DataLoader
from monai.metrics import ROCAUCMetric
import pandas as pd
from torchvision import datasets,transforms,models
import torchvision.transforms as datasets
import torch.nn.functional as F
from torchvision.transforms import functional as S
import torch.nn as nn
from tqdm import tqdm

from utils.config import *
from utils.config_test import *
from utils.file_sorting_test import *
from utils.file_sorting import *
from utils.Segmentation_model import *
from utils.CG_SSP import *
from utils.Dataloader import *
from utils.Dataloader_test import *
from utils.GCNT import *
from utils.MedCAM_OsteoCls_model import *
from utils.XMRCA import *
#from utils.metrics_test import *


checkpoint = torch.load(cfg.save_folder)

# Restore the model state
model.load_state_dict(checkpoint['model_state_dict'])

# Restore the optimizer state
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

# Restore the epoch and loss, if needed
epoch = checkpoint['epoch']
loss = checkpoint['loss']
weights = torch.tensor(cls_weights).to(cfg.device)
loss_function = torch.nn.CrossEntropyLoss(weight=weights) 
act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=cfg.num_classes)# n_classes=num_class

y_true = list()
y_predicted = list()
cross_entropy_list = list()
auc_metric = ROCAUCMetric()
features = []
w1=[]
w2 = []

with torch.no_grad():
    model.eval()
    y_pred = torch.tensor([], dtype=torch.float32, device = cfg.device)
    y = torch.tensor([], dtype=torch.long, device=cfg.device)
    
    for k, test_data in tqdm(enumerate(test_loader)):
        test_images1, test_images2,test_labels,image_name = test_data[0].to(cfg.device), test_data[1].to(cfg.device), test_data[2].to(cfg.device), test_data[3].to(cfg.device)
        outputs,feature = model(test_images1.float(),test_images2.float(), image_name)
        outputs1 = outputs.argmax(dim=1)
        y_pred = torch.cat([y_pred, outputs], dim=0)
        y = torch.cat([y, test_labels], dim=0)
        cross_entropy_f1 = [act(i) for i in y_pred]
        features.append(feature.cpu().detach().numpy().reshape(-1))
        #w1.append(weightage1)#cpu().detach().numpy())

        for i in y_pred:
           cs = act(i)
           cs1 = torch.max(cs)
           cs1 = round(cs1.item(),3)
        cross_entropy_list.append(cs1)
        for i in range(len(outputs)):
            y_predicted.append(outputs1[i].item())
            y_true.append(test_labels[i].item())
    y_onehot = [to_onehot(i) for i in y]
    y_pred_act = [act(i) for i in y_pred]
    auc_metric(y_pred_act, y_onehot)
    auc_result = auc_metric.aggregate()
    test_features = np.array(features)
    
    #print(len(w))
  
#saving the confusion metrics       
dict1 = {"image_name":M_test_image_file_list, "value":cross_entropy_list, 'y_true':y_true, 'y_predicted': y_predicted}
print(len(M_test_image_file_list),len(cross_entropy_list))
dt= pd.DataFrame(dict1)
dt.to_csv(cfg.folder_path+ "/" + cfg.model_name +".csv") 
save(test_cfg.embeddings_path,test_features)

file_path = cfg.folder_path+ "/" + cfg.model_name +".csv"

df = pd.read_csv(file_path) 
confusion_matrix = pd.crosstab(df['y_true'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, cmap="crest", annot=True, fmt=".0f")
plt.savefig(cfg.save_folder2)

#Calculate the QWK
from torchmetrics import ConfusionMatrix
def plot_confusion_matrix(y_true, y_pred, num_classes):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    cm = ConfusionMatrix(num_classes=num_classes, task='multiclass')
    cm.update(y_pred, y_true)
    cm_matrix = cm.compute().detach().cpu().numpy()
    return cm_matrix
    
confusion_matrix = plot_confusion_matrix(y_true, y_predicted, cfg.num_classes) 
   
y_true= df['y_true'].astype(int).tolist()
y_predicted = df['y_predicted'].astype(int).tolist()

from sklearn.metrics import cohen_kappa_score
y_true1 = np.array(y_true)
y_pred1 = np.array(y_predicted)
QWK = cohen_kappa_score(y_true1, y_pred1, weights="quadratic")


#Calculate MCC
def calculate_mcc(confusion_matrix):
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator
    mcc = np.mean(mcc)
    return mcc

mcc = calculate_mcc(confusion_matrix)

#Calculate MAE

def compute_mae(y_true, y_pred):
      y_true = torch.tensor(y_true, dtype=torch.float)
      y_pred = torch.tensor(y_pred, dtype=torch.float)
      mae = torch.mean(torch.abs(y_pred - y_true))
      return mae.item()

mae = compute_mae(y_true, y_predicted)
     
sys.stdout = open(cfg.save_folder3, "w")
print(classification_report(y_true, y_predicted, target_names=test_class_names, digits=4))
print("AUC:",auc_result)
print("QWK:", QWK)
print("MCC:", mcc)
print("MAE:",mae)

#Save tSNE Plot
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
embeddings = np.load(cfg.embeddings_path)
tsne_embeddings = tsne.fit_transform(embeddings)
test_predictions = np.array(y_predicted)
cmap = cm.get_cmap("Set1") #tab20
fig, ax = plt.subplots(figsize=(8,8))
plt.rcParams['axes.facecolor'] = 'white'
num_categories = 2
colors = ['cyan', 'green', 'red', 'yellow', 'black']
x_interval = 5  # Adjust this value based on your preference
y_interval = 5  # Adjust this value based on your preference

for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(tsne_embeddings[indices,0],tsne_embeddings[indices,1], c=colors[lab], label = "RKOA_Grade_"+ str(lab), edgecolors='black', alpha=0.7)
ax.legend(fontsize='large', markerscale=2)
plt.gca().set_aspect('equal')

plt.xlim(-45, 45)  # Replace with your desired range
plt.ylim(-45, 45)  # Replace with your desired range
plt.grid(True, color = 'gray',linestyle='-', linewidth=0.5)
plt.savefig(test_cfg.embeddings_path_png, dpi=400)

#Calculate Silhoutte score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import numpy as np

# X_tsne: The 2D embeddings from t-SNE
# labels: The ground truth labels for your data
X_test_image_label_list = np.array(X_test_image_label_list)

score = silhouette_score(tsne_embeddings, X_test_image_label_list)
print(f"Silhouette Score: {score}")


db_index = davies_bouldin_score(tsne_embeddings, X_test_image_label_list)
print(f"Davies-Bouldin Index: {db_index}")
