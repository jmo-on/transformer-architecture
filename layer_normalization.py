import torch
import torch.nn as nn

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