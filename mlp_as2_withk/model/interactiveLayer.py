
import torch
from torch import nn, einsum

from einops import rearrange

# functions

def exists(val):
    return val is not None
class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        init_eps = 1e-3,
    ):
        super().__init__()
        dim_out = dim
        self.norm = nn.LayerNorm(dim_out)
        #self.act = nn.Tanh()
        self.act = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.2)
        # parameters


        shape = (dim_out,dim_out)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_out
        #初始化函数
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        #self.bias = nn.Parameter(torch.ones(heads, dim_out))

    def forward(self, x,y):
        #device, n = x.device, x.shape[1]
        #https://blog.csdn.net/zouxiaolv/article/details/106371684
        x = x
        y = y
        x = self.norm(x)
        y = self.norm(y)

        weight = self.weight



        #gate = rearrange(x, 'b n (d) -> b n d')
        y = rearrange(y,'b n d -> b d n')
        gate = einsum('b l d,d d -> b l d', x, weight)
        gate = einsum('b l d,b d n -> b l n',gate, y)
        #gate = einsum('b n m, a b, b m n -> b n n', x, weight, y)
        gate = self.act(gate)
        #print(x.size())
        #print(gate.size())
        x = einsum('b n d,b n m -> b n d',x,gate)
        #print(x.size())
        #gate = rearrange(gate, 'b h n d -> b n (h d)')
        x = self.drop(x)
        return x

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.sgu = SpatialGatingUnit(dim_ff, seq_len)
        self.proj_out = nn.Sequential(
            nn.Linear(dim_ff, dim),
            nn.Dropout(p=0.2)
        )

    def forward(self, input):
        x,y = input
        x = self.proj_in(x)
        y = self.proj_in(y)

        x = self.sgu(x,y)
        y = self.sgu(y,x)

        x = self.proj_out(x)
        y = self.proj_out(y)

        return (x,y)

# main classes

class interactive(nn.Module):
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
            *[gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=seq_len) for _ in range(layer_num)]
        )

    def forward(self, x, y):

        x2,y2 = self.model((x,y))
        return x2, y2
