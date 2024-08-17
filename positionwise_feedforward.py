import torch.nn as nn

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