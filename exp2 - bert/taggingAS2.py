import os
import json
import argparse
import logging
import urllib
import sys
from tqdm import tqdm
from pycorenlp import StanfordCoreNLP
#from stanfordcorenlp import StanfordCoreNLP
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory to store tagging results.")
    parser.add_argument("--train_file", default='./data/WikiQA/wikicTrain.json', type=str,
                        help="WikiQA json for training. E.g., wikicTrain.json")
    parser.add_argument("--predict_file", default='./data/WikiQA/wikicDev.json', type=str,
                        help="WikiQA json for predictions. E.g., wikicDev.json or test-v1.1.json")
    parser.add_argument("--test_file", default='./data/WikiQA/wikicTest.json', type=str,
                        help="WikiQA json for predictions. E.g., wikicDev.json or test-v1.1.json")
    return parser.parse_args()


# transform corenlp tagging output into entity list
# some questions begins with whitespaces and they are striped by corenlp, thus begin offset should be added.

def parse_output(text, tagging_output, begin_offset=0, entitiesSet=None):
    entities = []
    select_states = ['ORGANIZATION', 'PERSON', 'MISC', 'LOCATION']
    for sent in tagging_output['sentences']:
        state = 'O'
        start_pos, end_pos = -1, -1
        for token in sent['tokens']:
            tag = token['ner']
            if tag == 'O' and state != 'O':
                if state in select_states:
                    entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos],
                                     'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                state = 'O'
            elif tag != 'O':
                #print(tag)
                entitiesSet.add(tag)
                #print(entitiesSet)
                if state == tag:
                    end_pos = token['characterOffsetEnd']
                else:
                    if state in select_states:
                        entities.append({'text': text[begin_offset + start_pos: begin_offset + end_pos],
                                         'start': begin_offset + start_pos, 'end': begin_offset + end_pos - 1})
                    state = tag
                    start_pos = token['characterOffsetBegin']
                    end_pos = token['characterOffsetEnd']
        if state in select_states:
            entities.append(
                {'text': text[begin_offset + start_pos: begin_offset + end_pos], 'start': begin_offset + start_pos,
                 'end': begin_offset + end_pos - 1})
    return entities,entitiesSet


def tagging(dataset, nlp, entitiesSet):
    skip_question_cnt, skip_answer_cnt = 0, 0
    k = 0
    entityCnt = 0
    for qas in dataset['data']:

            question = qas['question']
            question_tagging_output = nlp.annotate(urllib.parse.quote(question),
                                                  properties={'annotators': 'ner', 'outputFormat': 'json'})
            # assert the question length is not changed
            if len(question.strip()) == question_tagging_output['sentences'][-1]['tokens'][-1]['characterOffsetEnd']:
                question_entities, entitiesSet = parse_output(question, question_tagging_output, len(question) - len(question.lstrip()), entitiesSet)
            else:
                question_entities = []
                skip_question_cnt += 1
                logger.info('Skipped question due to offset mismatch:')
                logger.info(question)
            qas['question_entities'] = question_entities

            answer = qas['answer']
            answer_tagging_output = nlp.annotate(urllib.parse.quote(answer),
                                                   properties={'annotators': 'ner', 'outputFormat': 'json'})
            # assert the answer length is not changed
            if len(answer.strip()) == answer_tagging_output['sentences'][-1]['tokens'][-1]['characterOffsetEnd']:
                answer_entities, entitiesSet = parse_output(answer, answer_tagging_output,
                                                 len(answer) - len(answer.lstrip()), entitiesSet)
            else:
                answer_entities = []
                skip_answer_cnt += 1
                logger.info('Skipped answer due to offset mismatch:')
                logger.info(answer)
            qas['answer_entities'] = answer_entities
            print(k)
            k+=1
            if len(question_entities)>0 or len(answer_entities)>0:
                entityCnt+=1
    logger.info('{} set的命名实体类型集合：{}'.format(name, entitiesSet))
    print(entitiesSet)
    logger.info('共有{}条数据包含实体信息'.format(entityCnt))
    logger.info('In total, {} question and {} answer are skipped...'.format(skip_question_cnt, skip_answer_cnt))


if __name__ == '__main__':
    args = parse_args()

    # make output directory if not exist
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # register corenlp server
    nlp = StanfordCoreNLP('http://localhost:9753')


    # load train and dev datasets
    ftrain = open(args.train_file, 'r', encoding='utf-8')
    trainset = json.load(ftrain)
    fdev = open(args.predict_file, 'r', encoding='utf-8')
    devset = json.load(fdev)
    ftest = open(args.test_file, 'r', encoding='utf-8')
    testset = json.load(ftest)

    #for dataset, path, name in zip((trainset, devset), (args.train_file, args.predict_file), ('train', 'dev')):
    for dataset, path, name in zip((trainset, devset, testset), (args.train_file, args.predict_file, args.test_file),
                                   ('train', 'dev', 'test')):
        entitiesSet = set()
        tagging(dataset, nlp, entitiesSet)
        output_path = os.path.join(args.output_dir, "{}.tagged.json".format(os.path.basename(path)[:-5]))
        json.dump(dataset, open(output_path, 'w', encoding='utf-8'))
        logger.info('Finished tagging {} set'.format(name))


