import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
    self.q_layer = nn.Linear(d_model , d_model)
    self.linear_layer = nn.Linear(d_model, d_model)
  
  def forward(self, x, y, mask=None):
    batch_size, sequence_length, d_model = x.size()
    kv = self.kv_layer(x)
    q = self.q_layer(y)
    kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
    q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
    kv = kv.permute(0, 2, 1, 3)
    q = q.permute(0, 2, 1, 3)
    k, v = kv.chunk(2, dim=-1)
    values, attention = self.scaled_dot_product(q, k, v, mask)
    values = values.reshape(batch_size, sequence_length, d_model)
    out = self.linear_layer(values)
    return out
  
  def scaled_dot_product(self, q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
  
  def create_mask(batch_size=1, num_heads=8, sequence_length=512):
    mask = torch.full([batch_size, num_heads, sequence_length, sequence_length] , float('-inf'))
    mask = torch.triu(mask, diagonal=1)