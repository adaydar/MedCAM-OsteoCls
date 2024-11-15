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
from utils.file_sorting import *
from utils.Segmentation_model import *
from utils.CG_SSP import *
from utils.Dataloader import *
from utils.GCNT import *
from utils.MedCAM_OsteoCls_model import *
from utils.XMRCA import *


optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
weights = torch.tensor(cls_weights).to(cfg.device)
loss_function = torch.nn.CrossEntropyLoss(weight=weights) 
act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=cfg.num_classes)# n_classes=num_class
 

val_interval = 1


def weighted_loss(outputs, labels):
    softmax_op = torch.nn.Softmax(1)
    prob_pred = softmax_op(outputs)

    def set_weights():
        init_weights = np.array([[1, 3],
                                 [3, 1]], dtype=np.float32)

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights
    cls_weights = set_weights()

    batch_num, class_num = outputs.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot)
    class_hot = torch.autograd.Variable(class_hot).cuda()

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num
    return loss


#Scheduler

class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return optimizer
          
      
#Training
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
epoch_loss_values_ev = list()
auc_metric = ROCAUCMetric()
metric_values = list()
metric_values_train = list()
lr_scheduler = LRScheduler(cfg.lr, cfg.lr_decay_epoch)

for epoch in range(0,cfg.max_epochs):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{cfg.max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    optimizer = lr_scheduler(optimizer, epoch) #First model

    for i,batch_data in tqdm(enumerate(train_loader)):
        y_pred_train = torch.tensor([], dtype=torch.float32, device=cfg.device)
        y_train = torch.tensor([], dtype=torch.long, device=cfg.device)
        step += 1
        #print(batch_data.shape)
        #print(batch_data[3].shape)
        inputs1, inputs2, labels, image_name = batch_data[0].to(cfg.device),batch_data[1].to(cfg.device), batch_data[2].to(cfg.device), batch_data[3].to(cfg.device)
        optimizer.zero_grad()
        outputs,_ = model(inputs1.float(), inputs2.float(), image_name)  
        loss1 = loss_function(outputs, labels)
        loss2 = weighted_loss(outputs, labels)
        loss = loss1+loss2
        y_pred_train = torch.cat([y_pred_train, outputs], dim=0)
        y_train = torch.cat([y_train, labels], dim=0)
        #output1 = torch.max(outputs)
        #print(output1)
        #loss2 = lossL1(output1, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)
        epoch_loss += loss.item()
        print(f"{step}/{len(train_dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_dataset) // train_loader.batch_size

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")   
    y_onehot_train = [to_onehot(i) for i in y_train]
    y_pred_act_train = [act(i) for i in y_pred_train]
    auc_metric(y_pred_act_train, y_onehot_train)
    auc_result_train = auc_metric.aggregate()
    auc_metric.reset()
    del y_pred_act_train, y_onehot_train
    metric_values_train.append(auc_result_train)
    acc_value_train = torch.eq(y_pred_train.argmax(dim=1), y_train)
    acc_metric_train = acc_value_train.sum().item() / len(acc_value_train)        
    print(f" current accuracy: {acc_metric_train:.4f}")    

    if (epoch + 1) % val_interval == 0:
        model.eval()
        epoch_loss_ev = 0
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=cfg.device)
            y = torch.tensor([], dtype=torch.long, device=cfg.device)
            for val_data in val_loader:
                val_images1, val_images2, val_labels, val_names = val_data[0].to(cfg.device),val_data[1].to(cfg.device), val_data[2].to(cfg.device), val_data[3].to(cfg.device)
                outputs,_ = model(val_images1.float(), val_images2.float(), val_names)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                ev_loss1 = loss_function(outputs, val_labels)
                ev_loss2 = weighted_loss(outputs, val_labels)
                ev_loss = ev_loss1 + ev_loss2
                epoch_loss_ev += ev_loss.item()

            epoch_loss_values_ev.append(epoch_loss_ev)   
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(auc_result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), cfg.save_folder)
                torch.save({
                           'epoch': best_metric_epoch,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss,
                           }, cfg.save_folder)
            print(f"saved new AUC Metric is: {best_metric_epoch}")               
            print(f" current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")
            
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
dict1 = {"Loss_Values": epoch_loss_values}
F = pd.DataFrame(dict1)
F.to_csv(cfg.folder_path+"/"+"loss1.csv")

#Visulization
plt.figure('train', (12,6))
plt.subplot(1,2,1)
plt.title("Epoch Average Loss")
x = [i+1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1,2,2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i+1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel('epoch')
plt.plot(x,y)
plt.savefig(cfg.save_folder1)

