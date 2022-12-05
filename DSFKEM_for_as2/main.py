from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
def dynamic_expand(dynamic_tensor, smaller_tensor):
    #assert len(dynamic_tensor.shape) > len(smaller_tensor.shape)
    memory_embs_zero = dynamic_tensor.zero_().float()
    smaller_tensor = torch.add(memory_embs_zero,smaller_tensor)
    return smaller_tensor

if __name__ == '__main__':
  list = [1,2]
  list = list[:5]
  print(len(list))