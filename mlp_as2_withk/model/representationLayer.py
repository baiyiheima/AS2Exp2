import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange



class representationLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
    ):
        super().__init__()
        dim_out = dim
        self.norm = nn.LayerNorm(dim)

        #激活函数act = nn.Tanh()
        self.act = nn.Tanh()
        #self.act = nn.Softmax(dim=1)

        self.drop = nn.Dropout(0.2)

        # parameters

        shape = (dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        #初始化函数
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(dim_seq))

    def forward(self, x):
        #https://blog.csdn.net/zouxiaolv/article/details/106371684
        #res, gate = x.chunk(2, dim = -1)
        res = x
        gate = x
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        gate = einsum('b n d, m n -> b m d', gate, weight)
        gate = gate + rearrange(bias, 'n -> () n ()')

        result = self.act(gate) * res
        result = self.drop(result)

        return result
class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len
    ):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )

        self.layer1 = representationLayer(dim_ff, seq_len)

        self.proj_out = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(dim_ff, dim),
            nn.Dropout(p=0.2)
        )
    def forward(self, x):
        x = self.proj_in(x)
        x = self.layer1(x)
        x = self.proj_out(x)
        return x

# main classes

class representation(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_ff,
            seq_len,
            layer_num,
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len)for _ in range(layer_num)]
        )


    def forward(self,x):
        return self.model(x)
