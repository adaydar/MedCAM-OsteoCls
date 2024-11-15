import torch
from torchvision import datasets,transforms,models
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.file_sorting import *


###### Segmentation model ########################################################################
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
        
       
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=0,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
class cv1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv1,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x
       
class cv2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv2,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5,stride=1,padding=2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
class cv3(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cv3,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7,stride=1,padding=3,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x  
"""# Channel and Spatial Attention"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class adapool(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(adapool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                               nn.ReLU(),
                               )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        out = avg_out
        return self.sigmoid(out)        

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
class MRFF(nn.Module):
    def __init__(self,in_c,out_c):
      super(MRFF, self).__init__()
      self.conb1 = cv1(in_c,in_c)
      self.conb2 = cv2(in_c,in_c)
      self.conb3 = cv3(in_c,in_c)
      self.Conv1 = conv_block(ch_in= in_c*3,ch_out=out_c)
      self.GAP = adapool(in_c,out_c)
      self.Cnv1 = conv_block(ch_in= in_c,ch_out=out_c)
      
    def forward(self,x):
        x_1 = self.conb1(x)
        x_2 = self.conb2(x)
        x_3 = self.conb3(x)
        y_12 = torch.cat((x_1,x_2,x_3),dim=1) 
        x1 = self.Conv1(y_12)
        k1 = self.GAP(x)
        x1k = k1*x1
        x_add_1 = self.Cnv1(x)
        x1 = x1k +  x_add_1     
        return x1
                
class MtRA_Unet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(MtRA_Unet,self).__init__()
              
        self.mp = nn.MaxPool2d(2,stride=2,padding=1)
        self.ap = nn.AvgPool2d(2,stride=2,padding=1)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        #self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_adjust1 = nn.Conv2d(512,512,kernel_size=2,stride=1,padding=1)
        self.Conv_adjust_s1 = nn.Conv2d(1,1,kernel_size=2,stride=1,padding=0)
        self.Conv_adjust_c1 = nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0)
        self.Conv_adjust2 = nn.Conv2d(256,256,kernel_size=2,stride=1,padding=1)
        self.Conv_adjust_s2 = nn.Conv2d(1,1,kernel_size=2,stride=1,padding=0)
        self.Conv_adjust_c2 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.Conv_adjust3 = nn.Conv2d(128,128,kernel_size=2,stride=1,padding=1)
        self.Conv_adjust_s3 = nn.Conv2d(1,1,kernel_size=2,stride=1,padding=0)
        self.Conv_adjust_c3 = nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0)
        self.Conv_adjust_s4 = nn.Conv2d(1,1,kernel_size=2,stride=1,padding=0)
        self.Conv_adjust_c4 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0)
        self.Conv_adjust4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=2)
        self.sa1 =  SpatialAttention(512)
        self.ca1 =  ChannelAttention(512)
        self.sa2 =  SpatialAttention(256)
        self.ca2 =  ChannelAttention(256)
        self.sa3 =  SpatialAttention(128)
        self.ca3 =  ChannelAttention(128)
        self.sa4 =  SpatialAttention(64)
        self.ca4 =  ChannelAttention(64)
        self.MRFF1 = MRFF(1,64)
        self.MRFF2 = MRFF(64,128)
        self.MRFF3 = MRFF(128,256)
        self.MRFF4 = MRFF(256,512)
        self.MRFF5 = MRFF(512,1024)



    def forward(self,x):
        # encoding path
        x1 = self.MRFF1(x)

        y_13 = self.mp(x1)
        y_14 = self.ap(x1)
        y_15 = torch.add(y_13,y_14)
        x2 = self.MRFF2(y_15)

        
        y_23 = self.mp(x2)
        y_24 = self.ap(x2)
        y_25 = torch.add(y_23,y_24)          
        x3 = self.MRFF3(y_25)
        #print(x3.shape)
        
        y_33 = self.mp(x3)
        y_34 = self.ap(x3)
        y_35 = torch.add(y_33,y_34)        
        x4 = self.MRFF4(y_35)
        #print(x4.shape)
        
        y_43 = self.mp(x4)
        y_44 = self.ap(x4)
        y_45 = torch.add(y_43,y_44)       
        x5 = self.MRFF5(y_45)      
        #print(x5.shape)

        # decoding + concat path
        
        d5 = self.Up5(x5)
        d5 = self.Conv_adjust1(d5)
        X5_1 = self.sa1(x4)
        X5_1 = self.Conv_adjust_s1(X5_1)
        x4 = X5_1 * x4
        X5_2 = self.ca1(x4) 
        x4 = X5_2 * x4
        #print(x4.shape)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        X4_1 = self.sa2(x3)
        X4_1 = self.Conv_adjust_s2(X4_1)
        x3 = X4_1 * x3
        X4_2 = self.ca2(x3) 
        X4_2 = self.Conv_adjust_c2(X4_2)
        x3 = X4_2 * x3
        #print(x3.shape)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Conv_adjust3(d3)
        X3_1 = self.sa3(x2)
        X3_1 = self.Conv_adjust_s3(X3_1)
        x2 = X3_1 * x2
        X3_2 = self.ca3(x2) 
        X3_2 = self.Conv_adjust_c3(X3_2)
        x2 = X3_2 * x2  
        #print(x2.shape)     
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        X2_1 = self.sa4(x1)
        X2_1 = self.Conv_adjust_s4(X2_1)
        X1 = X2_1 * x1
        X2_2 = self.ca4(x1) 
        X2_2 = self.Conv_adjust_c4(X2_2)
        #print(X2_2.shape)
        x1 = X2_2 * x1
        #print(x1.shape)
        #print(d2.shape)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1, x4, x3, x2, x1

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    seg_model.load_state_dict(checkpoint["model_state_dict"])

                        
seg_model = MtRA_Unet()
load_checkpoint(torch.load(cfg.seg_dir), seg_model)

def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False

freeze_weights(seg_model)
