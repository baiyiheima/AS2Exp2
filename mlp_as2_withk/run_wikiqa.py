
import argparse
from utils.data import read_data,read_data_for_predict
from model.bert import BertClassfication
from utils.eval import eval_model
import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import BertModel,BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from reader_twomemory import DataProcessor

from utils.args import ArgumentGroup, print_arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("pre_trained_model",str,"bert-base-uncased","Init hugging face's model")
model_g.add_arg("dim",int,768,"Init hugging face's model")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 16, "Number of epoches for fine-tuning")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file", str, "data/WikiQA/json/wikic_train.json", "")
data_g.add_arg("predict_file", str, "data/WikiQA/json/wikic_test.json", "")
data_g.add_arg("raw_predict_file", str, "data/WikiQA/raw/wikic_test.tsv", "")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training.")
data_g.add_arg("max_seq_len", int, 64, "Number of words of the longest seqence.")
data_g.add_arg("max_question_len", int, 64, "Number of words of the longest seqence.")
data_g.add_arg("max_answer_len", int, 64, "Number of words of the longest seqence.")

mem_settings_g = ArgumentGroup(parser, "memory", "memory settings.")
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

    cn_id2concept, cn_concept2id, cn_concept_embedding_mat = read_concept_embedding(
        args.conceptnet_embedding_path)
    cn_concept_embedding_mat = torch.FloatTensor(cn_concept_embedding_mat)

    data_processer = DataProcessor(args)

    train_concept_settings = {
        'tokenization_path': '../data/WikiQA/tokenization/tokens/wikic.train.tokenization.uncased.data',
        'cn_concept2id':cn_concept2id,
        'use_conceptnet':args.use_conceptnet,
        'retrieved_conceptnet_path':args.retrieved_conceptnet_path,
    }
    batch_train_inputs = data_processer.data_generator(data_path=args.train_file,
                                                       batch_size=args.batch_size,
                                                       phase='train',
                                                       **train_concept_settings)

    predict_concept_settings = {
        'tokenization_path': '../data/WikiQA/tokenization/tokens/wikic.test.tokenization.uncased.data',
        'cn_concept2id': cn_concept2id,
        'use_conceptnet': args.use_conceptnet,
        'retrieved_conceptnet_path': args.retrieved_conceptnet_path,
    }
    batch_predict_inputs = data_processer.data_generator(data_path=args.predict_file,
                                                       batch_size=args.batch_size,
                                                       phase='predict',
                                                       **predict_concept_settings)

    model = BertClassfication(args,cn_concept_embedding_mat=cn_concept_embedding_mat,max_concept_size=data_processer.train_cn_max_concept_length)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if torch.cuda.is_available():
        model.cuda()
    lossfuction = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    epoch = args.epoch
    print_every_batch = 10
    #tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model)
    for k in range(epoch):
        model.train()
        print_avg_loss = 0
        for i in range(len(batch_train_inputs)):
            inputs = batch_train_inputs[i]
            targets = torch.tensor(inputs['label'])
            if torch.cuda.is_available():
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossfuction(outputs, targets)
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()
            if i % print_every_batch == (print_every_batch - 1):
                print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        logger.info("开始第 {} 轮的预测".format(k))
        map,mrr = eval_model( model, args.raw_predict_file, batch_predict_inputs)
        logger.info("{}* MAP: {}* MRR: {}\n".format(args.predict_file,map, mrr))
        with open('output/wikiqa_result.txt', mode='a+', encoding='utf-8') as file_obj:
                file_obj.write("Final Eval performance:\n* MAP: {}\n* MRR: {}\n".format(map, mrr))
if __name__ == '__main__':
    print_arguments(args)
    train(args)