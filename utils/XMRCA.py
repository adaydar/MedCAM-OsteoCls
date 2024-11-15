import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from utils.config import *

class XMRCA(nn.Module):
    def __init__(self, num_in_ch=32, dim=32, num_heads=8, bias=False):
        super(XMRCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(num_in_ch, dim, bias=bias)
        self.kv = nn.Linear(num_in_ch, dim * 2, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        self.residual_conv = nn.Linear(num_in_ch, dim, bias=bias)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x, y):
        
        b, l1 = x.shape  # batch, length, channels
        #print(x.shape)
        x = x.view(b,l1//32,l1//32)
        y = y.view(b,l1//32,l1//32)
        b,l,c = x.shape
        q = self.q(x)
        kv = self.kv(y)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b l (head c) -> b head l c', head=self.num_heads)
        k = rearrange(k, 'b l (head c) -> b head l c', head=self.num_heads)
        v = rearrange(v, 'b l (head c) -> b head l c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head l c -> b l (head c)', head=self.num_heads)
        #print(out.shape)
        out = self.project_out(out)
        #gc = self.global_context(out)
        #out = out * gc
        out = out + self.residual_conv(x)
        out = self.layer_norm(out)
        b,g,h = out.shape
        out = out.view(b,g*h)
        return out
        
cross_attn = XMRCA().to(cfg.device)



