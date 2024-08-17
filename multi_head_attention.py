import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    # Learnable layer to generate QKV
    self.qkv_layer = nn.Linear(d_model , 3 * d_model)
    # Multi-head layer to mix all heads
    self.linear_layer = nn.Linear(d_model, d_model)
  
  def forward(self, x, mask=None):
    batch_size, sequence_length, input_dim = x.size()
    qkv = self.qkv_layer(x)
    qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
    qkv = qkv.permute(0, 2, 1, 3)
    q, k, v = qkv.chunk(3, dim=-1)
    val, attention = self.scaled_dot_product(q, k, v, mask)
    val = val.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
    new_val = self.linear_layer(val)
    return new_val
  
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