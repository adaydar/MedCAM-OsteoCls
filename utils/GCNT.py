import torch
from torchvision import datasets,transforms,models
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import softmax, grid
import torchvision.transforms as T
from utils.config import *

############# GCNT_model ###################################################################        

class SelfAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        torch.nn.init.xavier_uniform_(self.att.data)

    def forward(self, x, edge_index):
        # Linear transformation
        H = self.lin(x)
        # Attention mechanism
        alpha = (H @ self.att).squeeze(-1)
        alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))
        return self.propagate(edge_index, x=H, alpha=alpha)

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNWithAttention, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.att1 = SelfAttentionLayer(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.att2 = SelfAttentionLayer(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        #print(x.shape)
        x = self.conv1(x, edge_index)
        x = self.att1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.att2(x, edge_index)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

Gcnn_mask = GCNWithAttention(in_channels = 10, hidden_channels = 48, out_channels=32).to(cfg.device)

  
class GCNN(nn.Module):
    def __init__(self, Gcnn_mask):
        super(GCNN, self).__init__()
        self.gcnn = Gcnn_mask

    def forward(self, x):
        #print(type(x))
        #print(x.shape)
        height, width = x.shape[1], x.shape[2]
        #batch_size = x.shape[0]
        num_channels = x.shape[0]
        x = x.view(num_channels, height * width) #batch_size, 
        _,edge_index = grid(height, width) 
        edge_index = edge_index.permute(1,0).to(cfg.device)
        edge_index = edge_index.to(torch.int64)
        #x = x.unsqueeze(0)
        x = x.permute(1,0)
        #print(type(edge_index))
        #print(type(x))
        out = self.gcnn(x.float(), edge_index)       
        return out
        
GCNN_mask = GCNN(Gcnn_mask).to(cfg.device)
checkpoint = torch.load(cfg.GCNN_dir)

# Restore the model state
GCNN_mask.load_state_dict(checkpoint['model_state_dict'])

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim))
        nn.init.uniform_(self.attention_weights)

    def forward(self, x):
        weights = torch.matmul(x, self.attention_weights)
        #weights = torch.softmax(weights, dim=0)
        x_weighted = x * weights.unsqueeze(-1)
        return x_weighted.sum(dim=0).mean().item()

attention_module = AttentionModule(32).to(cfg.device)

