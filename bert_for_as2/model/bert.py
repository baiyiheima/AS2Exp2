from transformers import BertModel,BertTokenizer
import torch.nn as nn
# 采用bert微调策略,在反向传播时一同调整BERT和线性层的参数，使bert更加适合分类任务
class BertClassfication(nn.Module):
    def __init__(self,args):
        super(BertClassfication, self).__init__()
        self.model_name = args.pre_trained_model
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, 2)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters
        self.softmax = nn.Softmax(dim=1)

    def forward(self, is_training, input_ids, token_type_ids, attention_mask):  # 这里的输入是一个list
        '''
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, pad_to_max_length=True)      #tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        '''
        hiden_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        outputs = hiden_outputs[0][:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        if not is_training:
            output = self.softmax(output)
        return output

