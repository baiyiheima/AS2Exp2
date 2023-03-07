import torch.nn as nn
from transformers import BertModel,BertTokenizer

class BERT(nn.Module):
    def __init__(self):
        super(BERT,self).__init__()
        self.model_name = args.pre_trained_model
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)