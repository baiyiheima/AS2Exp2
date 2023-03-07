import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.fn(self.norm(x))
        return x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, dim, seq_len,depth, expansion_factor = 4, dropout = 0.):
    chan_first, chan_last = nn.Linear, nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_first))
            #PreNormResidual(seq_len, FeedForward(seq_len, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)]
    )

'''
if __name__ == '__main__':

    model = MLPMixer(
        dim = 1068,
        seq_len=100,
        depth = 1
    )

    img = torch.randn(32, 100,1068)
    pred = model(img) # (1, 1000)
    print(pred.size())
'''