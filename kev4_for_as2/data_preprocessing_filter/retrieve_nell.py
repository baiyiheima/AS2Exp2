import pickle
import logging
import string
import argparse
import os
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from collections import namedtuple
from tqdm import tqdm
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--train_token', type=str,
                    default='../data/WikiQA/tokenization/tokens/wikic.train.tokenization.uncased.data',
                    help='token file of train set')
parser.add_argument('--dev_token', type=str,
                    default='../data/WikiQA/tokenization/tokens/wikic.dev.tokenization.uncased.data',
                    help='token file of dev set')
parser.add_argument('--test_token', type=str,
                    default='../data/WikiQA/tokenization/tokens/wikic.test.tokenization.uncased.data',
                    help='token file of dev set')
parser.add_argument('--score_threshold', type=float, default=0.90,
                    help='only keep generalizations relations with score >= threshold')
parser.add_argument('--output_dir', type=str, default='./output_wikiqa/', help='output directory')
args = parser.parse_args()

# remove category part of NELL entities, digit prefix 'n' and additional '_'
def preprocess_nell_ent_name(raw_name):
  ent_name = raw_name.split(':')[-1]
  digits = set(string.digits)
  if ent_name.startswith('n') and all([char in digits for char in ent_name.split('_')[0][1:]]):
      ent_name = ent_name[1:]
  ent_name = "_".join(filter(lambda x:len(x) > 0, ent_name.split('_')))
  return ent_name

if __name__ == '__main__':

    # load set of concepts with pre-trained embedding
    concept_set = set()
    with open('./nell_concept_list.txt') as fin:
        for line in fin:
            concept_name = line.strip()
            concept_set.add(concept_name)

    # read nell csv file and build NELL entity to category dict
    fin = open('./NELL.08m.1115.esv.csv',encoding='utf-8')
    nell_ent_to_cpt = {}
    nell_ent_to_fullname = {}

    header = True
    for line in tqdm(fin):
        if header:
            header = False
            continue
        line = line.strip()
        items = line.split('\t')
        if items[1] == 'generalizations' and float(items[4]) >= args.score_threshold:
            nell_ent_name = preprocess_nell_ent_name(items[0])
            category = items[2]
            if nell_ent_name not in nell_ent_to_cpt:
                nell_ent_to_cpt[nell_ent_name] = set()
                nell_ent_to_fullname[nell_ent_name] = set()
            nell_ent_to_cpt[nell_ent_name].add(category)
            nell_ent_to_fullname[nell_ent_name].add(items[0])
