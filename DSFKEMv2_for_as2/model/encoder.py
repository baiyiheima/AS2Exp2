# encoder.py

from torch import nn
import copy
from model.sublayer import LayerNorm, SublayerOutput

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    '''
    Transformer Encoder
    
    It is a stack of N layers.
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask, prior_att):
        for layer in self.layers:
            x,prior_att = layer(x, mask,prior_att)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    '''
    An encoder layer
    
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x, y, prior_att, mask=None):
        "Transformer Encoder"
        x, prior_att = self.sublayer_output[0](x, lambda x: self.self_attn(x, y, x, prior_att, mask )) # Encoder self-attention
        x, _ = self.sublayer_output[1](x, self.feed_forward)
        return x, prior_att