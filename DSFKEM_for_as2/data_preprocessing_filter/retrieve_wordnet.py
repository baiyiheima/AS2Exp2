import pickle
import argparse
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import logging
import string
from tqdm import tqdm
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--train_token', type=str, default='../data/WikiQA/tokenization/tokens/wikic.train.tokenization.uncased.data', help='token file of train set')
parser.add_argument('--dev_token', type=str, default='../data/WikiQA/tokenization/tokens/wikic.dev.tokenization.uncased.data', help='token file of dev set')
parser.add_argument('--test_token', type=str, default='../data/WikiQA/tokenization/tokens/wikic.test.tokenization.uncased.data', help='token file of dev set')
parser.add_argument('--output_dir', type=str, default='../output_wikiqa/', help='output directory')
parser.add_argument('--no_stopwords', action='store_true', help='ignore stopwords')
parser.add_argument('--ignore_length', type=int, default=0, help='ignore words with length <= ignore_length')
args = parser.parse_args()

if __name__ == '__main__':
    # initialize mapping from offset id to wn18 synset name
    offset_to_wn18name_dict = {}
    fin = open('./wordnet-mlj12-definitions.txt')
    for line in fin:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        offset_to_wn18name_dict[offset_str] = synset_name

    # load pickled samples
    train_samples = pickle.load(open(args.train_token, 'rb'))
    dev_samples = pickle.load(open(args.dev_token, 'rb'))
    test_samples = pickle.load(open(args.test_token, 'rb'))

    all_token_set = set()
    for sample in train_samples + dev_samples + test_samples:
        for token in sample['question_tokens'] + sample['answer_tokens']:
            all_token_set.add(token)

    # load stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # retrive synsets
    token2synset = dict()
    stopword_cnt = 0
    punctuation_cnt = 0
    for token in tqdm(all_token_set):
        # print(token)
        if token in set(string.punctuation):
            print('{} is punctuation, skipped!'.format(token))
            punctuation_cnt += 1
            continue
        if args.no_stopwords and token in stopwords:
            print('{} is stopword, skipped!'.format(token))
            stopword_cnt += 1
            continue
        if args.ignore_length > 0 and len(token) <= args.ignore_length:
            print('{} is too short, skipped!'.format(token))
            continue
        synsets = wn.synsets(token)
        wn18synset_names = []
        for synset in synsets:
            # print(synset)
            offset_str = str(synset.offset()).zfill(8)
            # print(offset_str)
            if offset_str in offset_to_wn18name_dict:
                wn18synset_names.append(offset_to_wn18name_dict[offset_str])
        # print(wn18synset_names)
        # print("==========")
        if len(wn18synset_names) > 0:
            token2synset[token] = wn18synset_names

        with open(os.path.join(args.output_dir, 'retrived_synsets.data'), 'wb') as fout:
            pickle.dump(token2synset, fout)