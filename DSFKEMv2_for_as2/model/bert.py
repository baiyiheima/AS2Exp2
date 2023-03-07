from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch
from model.layer import MemoryLayer,TriLinearTwoTimeSelfAttentionLayer

from model.encoder import EncoderLayer,Encoder
from model.feed_forward import PositionwiseFeedForward
from model.attention import MultiHeadedAttention
from copy import deepcopy

# 采用bert微调策略,在反向传播时一同调整BERT和线性层的参数，使bert更加适合分类任务
class BertClassfication(nn.Module):
    def __init__(self,args,wn_concept_embedding_mat,nell_concept_embedding_mat):
        super(BertClassfication, self).__init__()

        self.args = args

        self.model_name = args.pre_trained_model
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(args.transformer_dim, 2)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters
        self.softmax = nn.Softmax(dim=1)

        # 2nd layer: Memory Layer
        self.wn_embedding = nn.Embedding.from_pretrained(embeddings=wn_concept_embedding_mat,freeze=True)
        self.nell_embedding = nn.Embedding.from_pretrained(embeddings=nell_concept_embedding_mat,freeze=True)
        self.wn_memory_layer = MemoryLayer(bert_size=768, mem_emb_size=wn_concept_embedding_mat.shape[1], mem_method='raw')
        self.nell_memory_layer = MemoryLayer(bert_size=768,mem_emb_size=nell_concept_embedding_mat.shape[1], mem_method='raw')

        #3nd layer: self-matching
        '''
        self.self_att_layer = TriLinearTwoTimeSelfAttentionLayer(
        968, dropout_rate=0.0,
        cat_mul=True, cat_sub=True, cat_twotime=True,
        cat_twotime_mul=False, cat_twotime_sub=True)  # [bs, sq, concat_hs]
        '''
        attn = MultiHeadedAttention(args.transformer_h, args.transformer_dim,args.max_seq_len)
        ff = PositionwiseFeedForward(args.transformer_dim, args.transformer_dim, 0.2)
        self.encoder = Encoder(EncoderLayer(args.transformer_dim, deepcopy(attn), deepcopy(ff), 0.2), args.transformer_n)

    def forward(self,
                is_training,
                input_ids,
                token_type_ids,
                attention_mask,
                wn_concept_ids,
                nell_concept_ids,
                max_wn_concept_length,
                max_nell_concept_length):  # 这里的输入是一个list
        '''
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, pad_to_max_length=True)      #tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        '''
        #1 Layer
        hiden_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        enc_out = hiden_outputs[0]

        #2 Layer
        wn_memory_embs = self.wn_embedding(wn_concept_ids)  #[batch_size, seq_size, wn_concept_size, mem_emb_size]
        nell_memory_embs = self.nell_embedding(nell_concept_ids)  #[batch_size, seq_size, wn_concept_size, mem_emb_size]

        wn_concept_ids_reduced = wn_concept_ids > 0  #[batch_size, seq_size, wn_concept_size]
        wn_concept_ids_reduced = wn_concept_ids_reduced.float()  #[batch_size, seq_size, wn_concept_size]
        wn_mem_length = wn_concept_ids_reduced.sum(dim=2)  #[batch_size, seq_size]
        wn_mem_length = torch.unsqueeze(wn_mem_length, 2)  #[batch_size, seq_size, 1]
        nell_concept_ids_reduced = nell_concept_ids > 0  #[batch_size, seq_size, nell_concept_size]
        nell_concept_ids_reduced = nell_concept_ids_reduced.float()  #[batch_size, seq_size, nell_concept_size]
        nell_mem_length = nell_concept_ids_reduced.sum(dim=2)  #[batch_size, seq_size]
        nell_mem_length = torch.unsqueeze(nell_mem_length, 2)  #[batch_size, seq_size, 1]

        wn_memory_output, wn_prior_att = self.wn_memory_layer(bert_output=enc_out, memory_embs=wn_memory_embs, mem_length=wn_mem_length,  ignore_no_memory_token="True", concept_size=max_wn_concept_length)
        # [batch_size, seq_size, mem_emb_size]
        nell_memory_output, nell_prior_att = self.nell_memory_layer(bert_output=enc_out, memory_embs=nell_memory_embs, mem_length=nell_mem_length,  ignore_no_memory_token="True", concept_size=max_nell_concept_length)
        # [batch_size, seq_size, mem_emb_size]
        memory_output = torch.cat((enc_out,wn_memory_output,nell_memory_output), dim=2)  #[batch_size, seq_size, bert_emb_size+2*mem_emb_size]


        #3 Layer
        #memory_output_size = memory_output.shape[2]
        '''
        attention_mask = torch.unsqueeze(attention_mask,2)
        att_out_output = self.self_att_layer(memory_output,attention_mask) #[bs,sq,concat_hs]
        '''
        prior_att = torch.add(wn_prior_att,nell_prior_att)
        prior_att = torch.unsqueeze(prior_att,1)
        prior_att = prior_att.repeat(1,self.args.transformer_h,1,1)
        att_out_output = self.encoder(memory_output, memory_output, prior_att)
        #4 Layer
        outputs = att_out_output[:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        if not is_training:
            output = self.softmax(output)
        return output

