import argparse
import logging
import json
import os
import pickle
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer
import tokenization

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument("--output_dir", default='../data/YahooQA/tokenization/tokens', type=str,
                    help="The output directory to dump tokenization results.")
parser.add_argument("--train_file",
                    default='../data/YahooQA/tagged/yahooqa_train.tagged.json', type=str,
                    help="")
parser.add_argument("--dev_file",
                    default='../data/YahooQA/tagged/yahooqa_dev.tagged.json', type=str,
                    help="")
parser.add_argument("--test_file",
                    default='../data/YahooQA/tagged/yahooqa_test.tagged.json', type=str,
                    help="")

args = parser.parse_args()


class QASExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 id,
                 question_text,
                 question_entities_strset,
                 answer_text,
                 answer_entities_strset,
                 label
                 ):
        self.id = id
        self.question_text = question_text
        self.question_entities_strset = question_entities_strset
        self.answer_text = answer_text
        self.answer_entities_strset = answer_entities_strset
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "id: %s " % (self.id)
        s += ", question_text: %s" % ((self.question_text))
        s += ", answer_text: %s" % ((self.answer_text))
        return s


# the tokenization process when reading examples
def read_qas_examples(input_file):
    """Read a qas json file into a list of qasExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        id = entry["id"]
        question_text = entry["question"]
        question_entities_strset = set([entity_info["text"] for entity_info in entry["question_entities"]])
        answer_text = entry["answer"]
        answer_entities_strset = set([entity_info["text"] for entity_info in entry["answer_entities"]])
        label = entry["label"]

        example = QASExample(
            id=id,
            question_text=question_text,
            question_entities_strset=question_entities_strset,
            answer_text=answer_text,
            answer_entities_strset=answer_entities_strset,
            label=label)
        examples.append(example)
    return examples


def _is_real_subspan(start, end, other_start, other_end):
    return (start >= other_start and end < other_end) or (start > other_start and end <= other_end)


def match_query_entities(query_tokens, entities_tokens):
    # transform query_tokens list into a whitespace separated string
    query_string = " ".join(query_tokens)
    offset_to_tid_map = []
    tid = 0
    for char in query_string:
        offset_to_tid_map.append(tid)
        if char == ' ':
            tid += 1

    # transform entity_tokens into whitespace separated strings
    entity_strings = set()
    for entity_tokens in entities_tokens:
        entity_strings.add(" ".join(entity_tokens))

    # do matching
    results = []
    for entity_string in entity_strings:
        start = 0
        while True:
            pos = query_string.find(entity_string, start)
            if pos == -1:
                break
            token_start, token_end = offset_to_tid_map[pos], offset_to_tid_map[pos] + entity_string.count(' ')
            # assure the match is not partial match (eg. "ville" matches to "danville")
            if " ".join(query_tokens[token_start: token_end + 1]) == entity_string:
                results.append((token_start, token_end))
            start = pos + len(entity_string)

    # filter out a result span if it's a subspan of another span
    no_subspan_results = []
    for result in results:
        if not any(
                [_is_real_subspan(result[0], result[1], other_result[0], other_result[1]) for other_result in results]):
            no_subspan_results.append((" ".join(query_tokens[result[0]: result[1] + 1]), result[0], result[1]))
    assert len(no_subspan_results) == len(set(no_subspan_results))

    return no_subspan_results
# the further tokenization process when generating features
def tokenization_on_examples(examples, tokenizer):
    tokenization_result = []
    k = 0
    for example in tqdm(examples):
        # do tokenization on raw question text
        question_subtokens = []
        question_sub_to_ori_index = []  # mapping from sub-token index to token index
        question_tokens = tokenizer.basic_tokenizer.tokenize(example.question_text)
        for index, token in enumerate(question_tokens):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                question_subtokens.append(sub_token)
                question_sub_to_ori_index.append(index)

        # do tokenization on raw answer text
        answer_subtokens = []
        answer_sub_to_ori_index = []  # mapping from sub-token index to token index
        answer_tokens = tokenizer.basic_tokenizer.tokenize(example.answer_text)
        for index, token in enumerate(answer_tokens):
            for sub_token in tokenizer.wordpiece_tokenizer.tokenize(token):
                answer_subtokens.append(sub_token)
                answer_sub_to_ori_index.append(index)

        # match question entities (including tagged and document entities)
        entities_tokens = []
        for question_entity_str in example.question_entities_strset:
            entities_tokens.append(tokenizer.basic_tokenizer.tokenize(question_entity_str))
        question_entities = match_query_entities(question_tokens, entities_tokens)  # [('trump', 10, 10)]

        # match answer entities (including tagged and document entities)
        entities_tokens = []
        for answer_entity_str in example.answer_entities_strset:
            entities_tokens.append(tokenizer.basic_tokenizer.tokenize(answer_entity_str))
        answer_entities = match_query_entities(answer_tokens, entities_tokens)  # [('trump', 10, 10)]

        '''
        if len(answer_entities) > 0:

          print(example.answer_text)
        #print(question_tokens)
        #print(question_subtokens)
          print(answer_entities)
        #print(question_sub_to_ori_index)
        #print("===============")
        '''
        tokenization_result.append({
            'id': example.id,
            'question_tokens': question_tokens,
            'question_subtokens': question_subtokens,
            'question_entities': question_entities,
            'question_sub_to_ori_index': question_sub_to_ori_index,
            'answer_tokens': answer_tokens,
            'answer_subtokens': answer_subtokens,
            'answer_entities': answer_entities,
            'answer_sub_to_ori_index': answer_sub_to_ori_index,
            'label': example.label
        })
        '''
        if len(answer_subtokens) > 64:
            k+=1
            print(tokenization_result[len(tokenization_result)-1])
        '''

    return tokenization_result

if __name__ == '__main__':

    #for do_lower_case in (True):
        do_lower_case = True

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if do_lower_case else 'bert-base-cased')

        train_examples = read_qas_examples(input_file=args.train_file)
        train_tokenization_result = tokenization_on_examples(
            examples=train_examples,
            tokenizer=bert_tokenizer)

        with open(os.path.join(args.output_dir,'yahooqa.train.tokenization.{}.data'.format('uncased' if do_lower_case else 'cased')),'wb') as fout:
            pickle.dump(train_tokenization_result, fout)


        dev_examples = read_qas_examples(input_file=args.dev_file)
        dev_tokenization_result = tokenization_on_examples(
            examples=dev_examples,
            tokenizer=bert_tokenizer)

        with open(os.path.join(args.output_dir,'yahooqa.dev.tokenization.{}.data'.format('uncased' if do_lower_case else 'cased')),'wb') as fout:
            pickle.dump(dev_tokenization_result, fout)


        test_examples = read_qas_examples(input_file=args.test_file)
        test_tokenization_result = tokenization_on_examples(
            examples=test_examples,
            tokenizer=bert_tokenizer)

        with open(os.path.join(args.output_dir,'yahooqa.test.tokenization.{}.data'.format('uncased' if do_lower_case else 'cased')),'wb') as fout:
            pickle.dump(test_tokenization_result, fout)
