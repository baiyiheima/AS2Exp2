import torch.nn as nn
from transformers import BertModel,BertTokenizer

class BERT(nn.Module):
    def __init__(self,args):
        super(BERT,self).__init__()
        self.model_name = args.pre_trained_model
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, 2)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters
        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask):
        hiden_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        enc_out = hiden_outputs[0]
        outputs = enc_out[:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        return output