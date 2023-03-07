import argparse
from model.bert import BERT
import torch
import torch.nn as nn
import numpy as np
import logging
from transformers import BertModel,BertTokenizer
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

args = parser.parse_args()


def train(args):
    model = BERT(args)
    if torch.cuda.is_available():
        model.cuda()
    lossfuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch = args.epoch
    print_every_batch = 10
    for k in range(epoch):
        model.train()
        print_avg_loss = 0