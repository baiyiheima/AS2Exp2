
import argparse
from utils.data import read_data,read_data_for_predict
from model.bert import BertClassfication
from utils.eval import eval_model
import torch
import torch.nn as nn
import logging
from transformers import BertModel,BertTokenizer

from utils.args import ArgumentGroup, print_arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("pre_trained_model",str,"bert-base-uncased","Init hugging face's model")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 5, "Number of epoches for fine-tuning")
train_g.add_arg("learning_rate", float, 2e-5, "Learning rate used to train with warmup.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file", str, "data/WikiQA/wikic_train.tsv", "")
data_g.add_arg("predict_file", str, "data/WikiQA/wikic_test.tsv", "")
data_g.add_arg("batch_size", int, 8, "Total examples' number in batch for training.")
data_g.add_arg("max_seq_len", int, 200, "Number of words of the longest seqence.")

args = parser.parse_args()

def train(args):
    train_qas, train_label = read_data(args.train_file)
    predict_qas, predict_label = read_data(args.predict_file)


    batch_count = int(len(train_qas) / args.batch_size)
    batch_train_inputs, batch_train_targets = [], []
    for i in range(batch_count):
        batch_train_inputs.append(train_qas[i * args.batch_size: (i + 1) * args.batch_size])
        batch_train_targets.append(train_label[i * args.batch_size: (i + 1) * args.batch_size])

    predict_batch_count = int(len(predict_qas) / args.batch_size)
    batch_predict_inputs, batch_predict_targets = [], []
    for i in range(predict_batch_count):
        batch_predict_inputs.append(predict_qas[i * args.batch_size: (i + 1) * args.batch_size])
        batch_predict_targets.append(predict_label[i * args.batch_size: (i + 1) * args.batch_size])

    model = BertClassfication(args)
    if torch.cuda.is_available():
        model.cuda()
    lossfuction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch = args.epoch
    batch_count = batch_count
    print_every_batch = 5
    tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model)
    for k in range(epoch):
        model.train()
        print_avg_loss = 0
        for i in range(batch_count):
            inputs = batch_train_inputs[i]
            batch_tokenized = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                          max_length=args.max_seq_len, padding='max_length', truncation=True)  # tokenize、add special token、pad
            input_ids = torch.tensor(batch_tokenized['input_ids'])
            attention_mask = torch.tensor(batch_tokenized['attention_mask'])
            targets = torch.tensor(batch_train_targets[i])
            targets = targets.type(torch.LongTensor)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(True, input_ids, attention_mask)
            loss = lossfuction(outputs, targets)
            loss.backward()
            optimizer.step()

            print_avg_loss += loss.item()
            if i % print_every_batch == (print_every_batch - 1):
                print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        logger.info("开始第 {} 轮的预测".format(k))
        map,mrr = eval_model(args, model, args.predict_file,predict_batch_count, batch_predict_inputs, tokenizer)
        logger.info("{}* MAP: {}* MRR: {}\n".format(args.predict_file,map, mrr))
        with open('output/wikiqa_result.txt', mode='a+', encoding='utf-8') as file_obj:
                file_obj.write("Final Eval performance:\n* MAP: {}\n* MRR: {}\n".format(map, mrr))
if __name__ == '__main__':
    print_arguments(args)
    train(args)