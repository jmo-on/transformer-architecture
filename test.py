import torch
from encoder import Encoder
from decoder import Decoder

d_model = 512 # Default demension for overall vectors
num_heads = 8 # For multi-head attention
drop_prob = 0.1 # Drop out to regularize the network
batch_size = 30 # Helps better gradient descent + Parallel programming (Propagates after 30 trainings)
max_sequence_length = 200 # Max number of words for a single input
ffn_hidden = 2048 # Expland 512 dim vector only for Feed Forward layers
num_layers = 5 # Number of entire Encoder layer

x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = encoder(x)

x = torch.randn( (batch_size, max_sequence_length, d_model) ) # input sentence positional encoded 
y = torch.randn( (batch_size, max_sequence_length, d_model) ) # output sentence positional encoded 
mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
mask = torch.triu(mask, diagonal=1)
decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = decoder(x, y, mask)