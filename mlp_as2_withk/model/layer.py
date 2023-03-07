import torch.nn as nn
from torch import einsum
import torch
import numpy as np
from model.encoder import EncoderLayer,Encoder
from model.feed_forward import PositionwiseFeedForward
from model.attention import MultiHeadedAttention
from copy import deepcopy
from einops.layers.torch import Rearrange, Reduce

def dynamic_expand(dynamic_tensor, smaller_tensor):
    #assert len(dynamic_tensor.shape) > len(smaller_tensor.shape)
    memory_embs_zero = dynamic_tensor.clone()
    memory_embs_zero = memory_embs_zero.zero_().float()
    smaller_tensor = torch.add(memory_embs_zero,smaller_tensor)
    return smaller_tensor


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=5, stride=1, rate=1, activation=nn.ReLU()):
        super(Conv, self).__init__()
        padding = int(rate * (ksize - 1) / 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.activation = activation

    def forward(self, x):
        raw = self.conv(x)
        out = self.activation(raw)
        return out
    
class MemoryLayer(nn.Module):
    def __init__(self, bert_size, mem_emb_size, mem_method,max_concept_size,max_seq_len):
        super(MemoryLayer,self).__init__()
        self.mem_emb_size = mem_emb_size
        self.mem_method = mem_method
        self.linear1 = nn.Linear(bert_size, mem_emb_size)
        self.sentinel = nn.Parameter(torch.zeros([mem_emb_size]))
        self.softmax = nn.Softmax(dim=2)

        shape = (mem_emb_size,max_concept_size+1)
        weight = torch.zeros(shape)
        self.weight1 = nn.Parameter(weight)
        # 初始化函数
        nn.init.xavier_uniform_(self.weight1)

        shape2 = (mem_emb_size, max_concept_size + 1)
        weight2 = torch.zeros(shape2)
        self.weight2 = nn.Parameter(weight2)
        # 初始化函数
        nn.init.xavier_uniform_(self.weight2)

        self.tanh = nn.Tanh()
        self.softmax2 = nn.Softmax(dim=2)

        self.self_coding = nn.Sequential(
            Conv(max_seq_len,max_seq_len),
            nn.MaxPool2d(kernel_size=2),
            Conv(max_seq_len, max_seq_len),
            nn.LayerNorm(144),
            Reduce('b n c d -> b n d','mean'),
            nn.Linear(144,300)
        )

    def forward(self, bert_output, memory_embs, mem_length,  ignore_no_memory_token, concept_size,cn_concept_weights):
        """
        :param bert_output: [batch_size, seq_size, bert_size]
        :param memory_embs: [batch_size, seq_size, concept_size, mem_emb_size]
        :param mem_length: [batch_size, sent_size, 1]
        :param cn_concept_weights:[batch_size, sent_size, concept_size]
        :return: 
        """


        expanded_bert = torch.unsqueeze(self.linear1(bert_output),1).repeat(1,bert_output.shape[1],1,1)  # [batch_size, seq_size,seq_size, mem_emb_size]

        extended_memory, memory_score, attention_weights= self.add_sentinel(expanded_bert, memory_embs, self.mem_emb_size)
        # extended_memory: [batch_size, seq_size, 1+concept_size, mem_emb_size]
        # memory_score: [batch_size, seq_size, 1+concept_size]
        # attention_weights: # [batch,seq_size, seq_len,1+concept_size]

        concept_ordinal = self.get_concept_oridinal(concept_size, memory_score)  # [bs,sq,1+cs]
        memory_reverse_mask_infinity = (mem_length.expand(mem_length.shape[0],mem_length.shape[1],1+concept_size) < concept_ordinal).float() * (-1e6)
        del concept_ordinal
        memory_score = torch.add(memory_score,memory_reverse_mask_infinity)  # [batch_size, seq_size, 1+concept_size]
        del memory_reverse_mask_infinity
        sentinel_weight = torch.zeros(memory_score.shape[0],memory_score.shape[1],1)
        if torch.cuda.is_available():
            sentinel_weight = sentinel_weight.cuda()
        cn_concept_weights = torch.cat((sentinel_weight,cn_concept_weights),2)
        del sentinel_weight

        memory_score = self.softmax(memory_score)
        cn_concept_weights = self.softmax(cn_concept_weights)

        p = 0.7
        memory_score = torch.add(torch.mul(memory_score,p),cn_concept_weights,alpha=1-p)
        memroy_score_top_k = torch.topk(input=memory_score,k=5,dim=2).values  # [batch_size,sqe_size,k]
        memroy_score_top_k = torch.unsqueeze(memroy_score_top_k[:,:,-1],2).repeat(1,1,concept_size+1)  # [batch_size,sqe_size]
        memory_score_select = (memory_score >= memroy_score_top_k).float()
        del memroy_score_top_k
        memory_score = torch.mul(memory_score,memory_score_select)
        del memory_score_select
        memory_att = self.softmax(memory_score) # [batch_size, seq_size, 1+concept_size]



        memory_att = torch.unsqueeze(memory_att,dim=3).repeat(1,1,1,concept_size+1) # [batch_size, seq_size,  1+concept_size,1]
        summ = torch.matmul(memory_att,extended_memory)  # [batch_size, seq_size,1+concept_size, mem_emb_size]
        del memory_att
        concext_weight = einsum('b s a m,n c -> b s n m',summ, self.weight1) + einsum('b s a m,n c -> b s n m',torch.matmul(torch.transpose(attention_weights,2,3),expanded_bert),self.weight2)
        concext_weight = self.softmax2(self.tanh(concext_weight))
        summ = torch.matmul(summ,concext_weight)
        del concext_weight
        summ = self.self_coding(summ)
        summ = torch.squeeze(input=summ,dim=2)
        
        if ignore_no_memory_token:
            expand = torch.zeros([1],dtype=torch.float)
            if torch.cuda.is_available():
                expand = expand.cuda()
            condition = dynamic_expand(mem_length,expand) < mem_length
            condition = condition.float()
            summ = summ * condition

        if self.mem_method == "raw":
            output = summ # [batch_size,seq_len,1+concept_size,mem_emb_size]
        return output
            
            
    def get_concept_oridinal(self, concept_size, memory_score):
        """
        :param concept_size:
        :param memory_score: [batch_size, seq_size, 1+concept_size]
        :return:
        """
        concept_ordinal = torch.Tensor(np.arange(start=0, stop=(1 + concept_size), step=1, dtype=np.float32)) # [1+cs]
        if torch.cuda.is_available():
            concept_ordinal = concept_ordinal.cuda()
        concept_ordinal = dynamic_expand(memory_score, concept_ordinal)  # [bs,sq,1+cs]
        return concept_ordinal

    def add_sentinel(self, expanded_bert, memory_embs, mem_emb_size):
        """
        :param expanded_bert: [batch_size, seq_size, seq_size, mem_emb_size]
        :param memory_embs: [batch_size, seq_size, concept_size, mem_emb_size]
        :param mem_emb_size:
        :return:
        """
        sentinel = self.sentinel
        memory_embs_squeeze = memory_embs[:,:,0:1,:] # [bs,sq,1,ms]
        sentinel = dynamic_expand(memory_embs_squeeze, sentinel)  # [bs,sq,1,ms]
        extended_memory = torch.transpose(torch.cat((sentinel,memory_embs),2),2,3)  # [batch_size, seq_size, mem_emb_size, 1+concept_size]
        #extended_memory = torch.transpose(extended_memory, 2, 3)  # [batch_size, seq_size, mem_emb_size, 1+concept_size]
        attention_wights = torch.matmul(expanded_bert,extended_memory) # [batch,seq_size, seq_len,1+concept_size]
        memory_score = torch.mean(attention_wights,2)  # [batch_size, seq_size, 1+concept_size]
        #memory_score = torch.mean(memory_score,2) # [batch_size, seq_size, 1+concept_size]
        #memory_score = torch.max(memory_score,2)
        #memory_score = torch.squeeze(memory_score, dim=2) # [batch_size, seq_size, 1+concept_size]
        extended_memory = torch.transpose(extended_memory,  2, 3)# [batch_size, seq_size, 1+concept_size, mem_emb_size]
        return extended_memory, memory_score,attention_wights


class TriLinearTwoTimeSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0,
                 cat_mul=False, cat_sub=False, cat_twotime=False, cat_twotime_mul=False, cat_twotime_sub=False):
        super(TriLinearTwoTimeSelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.cat_mul = cat_mul
        self.cat_sub = cat_sub
        self.cat_twotime = cat_twotime
        self.cat_twotime_mul = cat_twotime_mul
        self.cat_twotime_sub = cat_twotime_sub

        self.bias = nn.Parameter(torch.zeros([1]).float())

        w1 = torch.empty(1, self.hidden_size).float()
        self.weight_1 = nn.Parameter(w1)
        nn.init.xavier_uniform_(self.weight_1)

        w2 = torch.empty(1, self.hidden_size).float()
        self.weight_2 = nn.Parameter(w2)
        nn.init.xavier_uniform_(self.weight_2)

        wmul = torch.empty(1, self.hidden_size).float()
        self.weight_mul = nn.Parameter(wmul)
        nn.init.xavier_uniform_(self.weight_mul)

        self.softmax = nn.Softmax(dim=2)

    def forward(self,hidden_emb, sequence_mask):
        """
        :param hidden_emb: [batch_size, seq_size, hidden_size]
        :param sequence_mask: [batch_size, seq_size, 1]
        :return:
        """
        assert len(hidden_emb.shape) == 3 and len(sequence_mask.shape) == 3 \
               and sequence_mask.shape[-1] == 1
        assert hidden_emb.shape[:2] == sequence_mask.shape[:2]

        hidden_size = self.hidden_size

        bs_1_hs = hidden_emb[:,0:1,:]  # [bs,1,hs]
        bs_hs_1 = torch.transpose(bs_1_hs, 1, 2)  # [bs,hs,1]
        weight_1 = dynamic_expand(bs_1_hs,self.weight_1)  # [bs,1,hs]
        weight_1 = torch.transpose(weight_1, 1, 2)
        hidden_emb_clone = hidden_emb.clone()
        r1 = torch.matmul(hidden_emb_clone,weight_1)  # [bs,sq,1]

        weight_2 = dynamic_expand(bs_1_hs,self.weight_2)  # [bs,1,hs]
        hidden_emb_transpose = torch.transpose(hidden_emb_clone,1,2)  # [bs,hs,sq]
        r2 = torch.matmul(weight_2,hidden_emb_transpose)  # [bs,1,sq]
        
        weight_mul = dynamic_expand(hidden_emb, self.weight_mul)
        rmul_1 = torch.mul(hidden_emb, weight_mul)  # [bs,sq,hs]
        rmul_2 = torch.matmul(rmul_1, hidden_emb_transpose)  # [bs,sq,sq]

        r1 = torch.squeeze(r1,dim=2)  # [bs, sq]
        r1 = dynamic_expand(torch.transpose(rmul_2,0,1), r1) # [sq,bs,sq]
        r1 = torch.transpose(r1,0,1) #[bs,sq,sq]

        r2 = torch.squeeze(r2,dim=1) # [bs,sq]
        r2 = dynamic_expand(torch.transpose(rmul_2,0,1), r2) # [sq,bs,sq]
        r2 = torch.transpose(r2,0,1) #[bs,sq,sq]

        bias = dynamic_expand(rmul_2,self.bias) # [bs,sq,sq]
        sim_score = torch.add(r1,r2)
        sim_score = torch.add(sim_score,rmul_2)
        sim_score = torch.add(sim_score,bias) #[bs,sq,sq]

        sequence_mask = sequence_mask.float()  # [bs,sq,1]
        softmax_mask = (sequence_mask-1)*(-1)
        very_negative_number = torch.tensor([-1e6]).float()
        if torch.cuda.is_available():
            very_negative_number = very_negative_number.cuda()
        softmax_mask = softmax_mask*very_negative_number  # [bs,sq,1]
        softmax_mask = torch.squeeze(softmax_mask,2)  #[bs,sq]
        softmax_mask = dynamic_expand(torch.transpose(torch.transpose(sim_score,0,1),0,2),softmax_mask)  # [sq,bs,sq]
        softmax_mask = torch.transpose(softmax_mask,0,1)  # [bs,sq,sq]
        sim_score = torch.add(sim_score,softmax_mask)  # [bs,sq,sq]

        attn_prob = self.softmax(sim_score) #[bs,sq,sq]
        weighted_sum = torch.matmul(attn_prob,hidden_emb)  # [bs,sq,sq]*[bs,sq,hs]=[BS,SQ,HS]
        if any([self.cat_twotime, self.cat_twotime_mul, self.cat_twotime_sub]):
            twotime_att_prob = torch.matmul(attn_prob, attn_prob)  # [bs,sq,sq]*[bs,sq,sq]=[BS,SQ,SQ]
            twotime_weited_sum = torch.matmul(twotime_att_prob, hidden_emb)  # [BS,SQ,HS]

        out_tensors = [hidden_emb, weighted_sum]
        if self.cat_mul:
            out_tensors.append(torch.mul(hidden_emb,weighted_sum))
        if self.cat_sub:
            out_tensors.append(torch.sub(hidden_emb,weighted_sum))
        if self.cat_twotime:
            out_tensors.append(twotime_weited_sum)
        if self.cat_twotime_mul:
            out_tensors.append(torch.mul(hidden_emb,twotime_weited_sum))
        if self.cat_twotime_sub:
            out_tensors.append(torch.sub(hidden_emb,twotime_weited_sum))
        output = torch.cat(out_tensors, dim=2)  # [BS,SQ, HS+HS+....]
        return output