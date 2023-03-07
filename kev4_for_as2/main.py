
import numpy as np
import torch
import torch.nn as nn
from model.encoder import EncoderLayer,Encoder
from model.feed_forward import PositionwiseFeedForward
from model.attention import MultiHeadedAttention
from copy import deepcopy

class Gated_Conv(nn.Module):
  def __init__(self, in_ch, out_ch, ksize=5, stride=1, rate=1, activation=nn.ReLU()):
    super(Gated_Conv, self).__init__()
    padding = int(rate * (ksize - 1) / 2)
    # 通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
    self.conv = nn.Conv2d(in_ch, 2 * out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
    self.activation = activation

  def forward(self, x):
    raw = self.conv(x)
    x1 = raw.split(int(raw.shape[1] / 2), dim=1)  # 将特征图分成两半，其中一半是做学习
    gate = torch.sigmoid(x1[0])  # 将值限制在0-1之间
    out = self.activation(x1[1]) * gate
    return out

if __name__ == '__main__':
  a = torch.randn(32, 100,64, 150)
  pool = nn.AdaptiveMaxPool2d((1,300))
  print(pool(a).size())


