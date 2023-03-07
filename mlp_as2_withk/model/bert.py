from transformers import BertModel,BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch
from model.representationLayer import representation
from model.interactiveLayer import interactive
from model.layer import MemoryLayer
from model.MLPMixer import MLPMixer

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class BertClassfication(nn.Module):
    def __init__(self,args,cn_concept_embedding_mat,max_concept_size):
        super(BertClassfication, self).__init__()
        self.model_name = args.pre_trained_model
        self.pre_model = BertModel.from_pretrained(self.model_name,output_hidden_states = True)
        freeze(self.pre_model)

        self.representation = representation(dim=args.dim, dim_ff=512, seq_len=args.max_question_len+2,layer_num=1)
        self.interactive = interactive(dim=args.dim, dim_ff=512, seq_len=args.max_question_len+2,layer_num=1)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        #knowledge enhance
        self.cn_embedding = nn.Embedding.from_pretrained(embeddings=cn_concept_embedding_mat, freeze=True)
        self.cn_memory_layer = MemoryLayer(bert_size=768, mem_emb_size=cn_concept_embedding_mat.shape[1],
                                           mem_method='raw', max_concept_size=max_concept_size, max_seq_len=max_seq_len)

        self.mlpmixer = MLPMixer(dim=1068, seq_len=max_seq_len, depth=1)

    def forward(self, inputs):  # 这里的输入是一个list

        q_input_ids = torch.tensor(inputs['q_input_ids'])
        q_attention_mask = torch.tensor(inputs['q_attention_mask'])
        a_input_ids = torch.tensor(inputs['a_input_ids'])
        a_attention_mask = torch.tensor(inputs['a_attention_mask'])
        q_cn_concept_ids = torch.tensor(inputs['q_cn_concept_ids'])
        q_cn_concept_weights = torch.tensor(inputs['q_cn_concept_weights'])
        a_cn_concept_ids = torch.tensor(inputs['a_cn_concept_ids'])
        a_cn_concept_weights = torch.tensor(inputs['a_cn_concept_weights'])
        
        if torch.cuda.is_available():
            q_input_ids = q_input_ids.cuda()
            q_attention_mask = q_attention_mask.cuda()
            a_input_ids = a_input_ids.cuda()
            a_attention_mask = a_attention_mask.cuda()
            q_cn_concept_ids = q_cn_concept_ids.cuda()
            q_cn_concept_weights = q_cn_concept_weights.cuda()
            a_cn_concept_ids = a_cn_concept_ids.cuda()
            a_cn_concept_weights = a_cn_concept_weights.cuda()
            
            
        self.pre_model.eval()
        with torch.no_grad():
            q_hiden_outputs = self.pre_model(q_input_ids, q_attention_mask)
            a_hiden_outputs = self.pre_model(a_input_ids,a_attention_mask)

        q_outputs = q_hiden_outputs[2]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        a_outputs = a_hiden_outputs[2]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果

        q_token_embeddings = torch.stack(q_outputs, dim=0)
        q_token_embeddings = q_token_embeddings[-4:,:,:,:]
        q_token_embeddings = torch.sum(q_token_embeddings,0)

        a_token_embeddings = torch.stack(a_outputs, dim=0)
        a_token_embeddings = a_token_embeddings[-4:, :, :, :]
        a_token_embeddings = torch.sum(a_token_embeddings, 0)

        q_cn_memory_embs = self.cn_embedding(q_cn_concept_ids)
        q_cn_concept_ids_reduced = (q_cn_concept_ids > 0).float()
        q_cn_mem_length = torch.unsqueeze(q_cn_concept_ids_reduced.sum(dim=2), 2)

        a_cn_memory_embs = self.cn_embedding(a_cn_concept_ids)
        a_cn_concept_ids_reduced = (a_cn_concept_ids > 0).float()
        a_cn_mem_length = torch.unsqueeze(a_cn_concept_ids_reduced.sum(dim=2), 2)

        q_cn_memory_output = self.cn_memory_layer(bert_output=q_token_embeddings, memory_embs=q_cn_memory_embs,
                                                mem_length=q_cn_mem_length, ignore_no_memory_token="True",
                                                concept_size=max_cn_concept_length,
                                                cn_concept_weights=q_cn_concept_weights)
        q_memory_output = torch.cat((q_token_embeddings, q_cn_memory_output), dim=2)

        a_cn_memory_output = self.cn_memory_layer(bert_output=a_token_embeddings, memory_embs=a_cn_memory_embs,
                                                  mem_length=a_cn_mem_length, ignore_no_memory_token="True",
                                                  concept_size=max_cn_concept_length,
                                                  cn_concept_weights=a_cn_concept_weights)
        a_memory_output = torch.cat((a_token_embeddings, a_cn_memory_output), dim=2)

        q_memory_output = self.mlpmixer(q_memory_output)
        a_memory_output = self.mlpmixer(a_memory_output)
        
        reQ = self.representation(q_memory_output)
        reA = self.representation(a_memory_output)
        inQ,inA = self.interactive(q_memory_output,a_memory_output)
        
        encodeQ = torch.cat((reQ,inQ),2)
        encodeA = torch.cat((reA, inA), 2)
        final_feature_Q = torch.mean(encodeQ, 1)
        final_feature_A = torch.mean(encodeA, 1)
        output = self.cos(final_feature_Q,final_feature_A)

        return output

