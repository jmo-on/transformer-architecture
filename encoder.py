import math
import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512 # Default demension for overall vectors
num_heads = 8 # For multi-head attention
drop_prob = 0.1 # Drop out to regularize the network
batch_size = 30 # Helps better gradient descent + Parallel programming (Propagates after 30 trainings)
max_sequence_length = 200 # Max number of words for a single input
ffn_hidden = 2048 # Expland 512 dim vector only for Feed Forward layers
num_layers = 5 # Number of entire Encoder layer
  
class MultiheadAttention(nn.Module):
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

class LayerNormalization(nn.Module):
  def __init__(self, parameters_shape, eps=1e-5):
    '''
    :param parameters_shape: [batch_size, input_dim]
    '''
    super().__init__()
    self.parameters_shape=parameters_shape
    self.eps=eps
    self.gamma = nn.Parameter(torch.ones(parameters_shape))
    self.beta =  nn.Parameter(torch.zeros(parameters_shape))

  def forward(self, x):
    '''
    :param x: input sequence size of [sequence_length, batch_size, input_dim]
    '''
    dims = [-(i + 1) for i in range(len(self.parameters_shape))]
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    y = self.gamma * (x - mean) / std  + self.beta
    return y
  
class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, ffn_hidden)
    self.linear2 = nn.Linear(ffn_hidden, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=drop_prob)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    return x
  
class EncoderLayer(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
    super(EncoderLayer, self).__init__()
    self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
    self.norm1 = LayerNormalization(parameters_shape=[d_model])
    self.dropout1 = nn.Dropout(p=drop_prob)
    self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
    self.norm2 = LayerNormalization(parameters_shape=[d_model])
    self.dropout2 = nn.Dropout(p=drop_prob)

  def forward(self, x):
    residual_x = x
    # Multi-head attention
    x = self.attention(x, mask=None)
    x = self.dropout1(x)
    # Add & Norm
    x = self.norm1(x + residual_x)
    residual_x = x
    # Feed Forward
    x = self.ffn(x)
    x = self.dropout2(x)
    # Add & Norm
    x = self.norm2(x + residual_x)
    return x

class Encoder(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
    super().__init__()
    self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

  def forward(self, x):
    x = self.layers(x)
    return x

''' Test '''
x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = encoder(x)