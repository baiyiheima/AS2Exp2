
import argparse
from utils.data import read_data,read_data_for_predict
from model.bert import BertClassfication
from utils.eval import eval_model
import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import BertModel,BertTokenizer
from reader_twomemory import DataProcessor
from tqdm import tqdm

from utils.args import ArgumentGroup, print_arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("pre_trained_model",str,"bert-base-uncased","Init hugging face's model")
model_g.add_arg("do_lower_case",bool,True,"")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 12, "Number of epoches for fine-tuning")
train_g.add_arg("learning_rate", float, 2e-5, "Learning rate used to train with warmup.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file", str, "../data/WikiQA/json/wikic_train.json", "")
data_g.add_arg("predict_file", str, "../data/WikiQA/json/wikic_test.json", "")
data_g.add_arg("raw_predict_file", str, "../data/WikiQA/raw/wikic_test.tsv", "")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training.")
data_g.add_arg("max_seq_len", int, 100, "Number of words of the longest seqence.")
data_g.add_arg("max_question_len", int, 20, "Number of words of the longest seqence.")
data_g.add_arg("max_answer_len", int, 77, "Number of words of the longest seqence.")


mem_settings_g = ArgumentGroup(parser, "memory", "memory settings.")
'''
mem_settings_g.add_arg('wn_concept_embedding_path',  str,    "../data/KB_embeddings/wn_concept2vec.txt",   'path of wordnet pretrained concept file')
mem_settings_g.add_arg(trieved_synset_path',   str,    './data/WikiQA/retrieve_wordnet/retrived_synsets.data',   'path of retrieved synsets')
mem_settings_g.add_arg('use_nell',                bool,   True,  'whether to use nell memory')
mem_settings_g.add_arg('train_ret'nell_concept_embedding_path',  str,   "../data/KB_embeddings/nell_concept2vec.txt",   'path of nell pretrained concept file')
mem_settings_g.add_arg('use_wordnet',             bool,   True,  'whether to use wordnet memory')
mem_settings_g.add_arg('rerieved_nell_concept_path',   str,    './data/WikiQA/retrieve_nell/train.retrieved_nell_concepts.data', 'path of retrieved concepts for trainset')
mem_settings_g.add_arg('pridect_retrieved_nell_concept_path',     str,    './data/WikiQA/retrieve_nell/test.retrieved_nell_concepts.data',   'path of retrieved concepts for devset')
'''
mem_settings_g.add_arg('conceptnet_embedding_path',  str,    "../data/KB_embeddings/numberbatch-en.txt",   'path of conceptnet pretrained concept file')
mem_settings_g.add_arg('use_conceptnet',             bool,   True,  'whether to use conceptnet memory')
mem_settings_g.add_arg('retrieved_conceptnet_path',   str,    '../data/WikiQA/retrieve_conceptnet/retrived_conceptnet_kbs_with_weight.data', '')

args = parser.parse_args()

def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[1].split(' ')[1:])
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    count = 0
    for line in tqdm(info):
        if count == 0:
            count += 1
            continue
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return id2concept, concept2id, embedding_mat

def train(args):

    '''
    wn_id2concept, wn_concept2id, wn_concept_embedding_mat = read_concept_embedding(
        args.wn_concept_embedding_path)
    nell_id2concept, nell_concept2id, nell_concept_embedding_mat = read_concept_embedding(
        args.nell_concept_embedding_path)
    wn_concept_embedding_mat = torch.FloatTensor(wn_concept_embedding_mat)
    nell_concept_embedding_mat = torch.FloatTensor(nell_concept_embedding_mat)
    '''
    cn_id2concept, cn_concept2id, cn_concept_embedding_mat = read_concept_embedding(
        args.conceptnet_embedding_path)
    cn_concept_embedding_mat = torch.FloatTensor(cn_concept_embedding_mat)


    data_processer = DataProcessor(do_lower_case=args.do_lower_case,
                                   max_seq_length=args.max_seq_len,
                                   max_question_length=args.max_question_len,
                                   max_answer_length=args.max_answer_len
                                   )
    train_concept_settings = {
        'tokenization_path': '../data/WikiQA/tokenization/tokens/wikic.train.tokenization.{}.data'.format(
            'uncased' if args.do_lower_case else 'cased'),
        'cn_concept2id':cn_concept2id,
        'use_conceptnet':args.use_conceptnet,
        'retrieved_conceptnet_path':args.retrieved_conceptnet_path,
    }
    batch_train_inputs = data_processer.data_generator(data_path=args.train_file,
                                                       batch_size=args.batch_size,
                                                       phase='train',
                                                       **train_concept_settings)

    predict_concept_settings = {
        'tokenization_path': '../data/WikiQA/tokenization/tokens/wikic.test.tokenization.{}.data'.format(
            'uncased' if args.do_lower_case else 'cased'),
        'cn_concept2id': cn_concept2id,
        'use_conceptnet': args.use_conceptnet,
        'retrieved_conceptnet_path': args.retrieved_conceptnet_path,
    }
    batch_predict_inputs = data_processer.data_generator(data_path=args.predict_file,
                                                       batch_size=args.batch_size,
                                                       phase='predict',
                                                       **predict_concept_settings)

    model = BertClassfication(args=args,
                              cn_concept_embedding_mat=cn_concept_embedding_mat,max_concept_size=data_processer.train_cn_max_concept_length,max_seq_len=args.max_seq_len)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if torch.cuda.is_available():
        model.cuda()
    lossfuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch = args.epoch
    print_every_batch = 10
    #tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model)
    for k in range(epoch):
        model.train()
        print_avg_loss = 0
        for i in range(len(batch_train_inputs)):
            inputs = batch_train_inputs[i]
            '''
            batch_tokenized = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                          max_length=args.max_seq_len, padding='max_length', truncation=True)  # tokenize、add special token、pad
            '''
            input_ids = torch.tensor(inputs['input_ids'])
            token_type_ids = torch.tensor(inputs['token_type_ids'])
            attention_mask = torch.tensor(inputs['attention_mask'])
            cn_concept_ids = torch.tensor(inputs["cn_concept_ids"])
            cn_concept_weights = torch.tensor(inputs["cn_concept_weights"])
            targets = torch.tensor(inputs['label'])
            targets = targets.type(torch.LongTensor)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                cn_concept_ids = cn_concept_ids.cuda()
                cn_concept_weights = cn_concept_weights.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(True,
                            input_ids,
                            token_type_ids,
                            attention_mask,
                            cn_concept_ids,
                            cn_concept_weights,
                            data_processer.train_cn_max_concept_length)
            loss = lossfuction(outputs, targets)
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()
            if i % print_every_batch == (print_every_batch - 1):
                print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        logger.info("开始第 {} 轮的预测".format(k))
        map,mrr = eval_model( model, args.raw_predict_file, batch_predict_inputs,data_processer)
        logger.info("{}* MAP: {}* MRR: {}\n".format(args.predict_file,map, mrr))
        with open('output/wikiqa_result.txt', mode='a+', encoding='utf-8') as file_obj:
                file_obj.write("Final Eval performance:\n* MAP: {}\n* MRR: {}\n".format(map, mrr))
if __name__ == '__main__':
    print_arguments(args)
    train(args)