import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from positionwise_feedforward import PositionwiseFeedForward
  
class EncoderLayer(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
    super(EncoderLayer, self).__init__()
    self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.norm1 = LayerNormalization(parameters_shape=[d_model])
    self.dropout1 = nn.Dropout(p=drop_prob)
    self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
    self.norm2 = LayerNormalization(parameters_shape=[d_model])
    self.dropout2 = nn.Dropout(p=drop_prob)

  def forward(self, x):
    _x = x
    # Multi-head attention
    x = self.attention(x, mask=None)
    x = self.dropout1(x)
    # Add & Norm
    x = self.norm1(x + _x)
    _x = x
    # Feed Forward
    x = self.ffn(x)
    x = self.dropout2(x)
    # Add & Norm
    x = self.norm2(x + _x)
    return x

class Encoder(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
    super().__init__()
    self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

  def forward(self, x):
    return self.layers(x)