import json
import os
import pickle
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from data_preprocessing import tokenization
from tqdm import tqdm


class QASExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
            id,
            question_text,
            answer_text,
            label
            ):
      self.id = id
      self.question_text = question_text
      self.answer_text = answer_text
      self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += ", question_text: %s" % (self.question_text)
        s += ", answer_text: %s" % (self.answer_text)
        return s
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 q_tokens,
                 q_input_ids,
                 q_input_mask,
                 q_segment_ids,
                 q_pos_ids,
                 q_cn_concept_ids,
                 q_cn_concept_weights,
                 a_tokens,
                 a_input_ids,
                 a_input_mask,
                 a_segment_ids,
                 a_pos_ids,
                 a_cn_concept_ids,
                 a_cn_concept_weights,
                 label):
        self.qas_id = qas_id
        self.q_tokens = q_tokens
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.q_pos_ids = q_pos_ids
        self.a_tokens = a_tokens
        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.a_pos_ids = a_pos_ids
        self.label = label
        self.q_cn_concept_ids = q_cn_concept_ids
        self.q_cn_concept_weights = q_cn_concept_weights
        self.a_cn_concept_ids = a_cn_concept_ids
        self.a_cn_concept_weights = a_cn_concept_weights

class Examples_To_Features_Converter(object):
    def __init__(self, **concept_settings):
        self.concept_settings = concept_settings

        tokenization_filepath = self.concept_settings['tokenization_path']
        assert os.path.exists(tokenization_filepath)
        self.all_tokenization_info = {}
        for item in pickle.load(open(tokenization_filepath, 'rb')):
            self.all_tokenization_info[item['id']] = item

        self.cn_concept2id = self.concept_settings['cn_concept2id']

        if concept_settings['use_conceptnet']:
            retrieved_conceptnet_filepath = self.concept_settings['retrieved_conceptnet_path']
            assert os.path.exists(retrieved_conceptnet_filepath)
            self.synsets_info = pickle.load(open(retrieved_conceptnet_filepath, 'rb'))  # token to sysnet names
            self.max_cn_concept_length = max([len(synsets) for synsets in self.synsets_info.values()])

    def _lookup_conceptnet_concept_ids(self, sub_tokens, sub_to_ori_index, tokens, tolower, tokenizer):
        concept_ids = []
        weights = []
        for index in range(len(sub_tokens)):
            original_token = tokens[sub_to_ori_index[index]]
            # if tokens are in upper case, we must lower it for retrieving
            retrieve_token = tokenizer.basic_tokenizer._run_strip_accents(
                original_token.lower()) if tolower else original_token
            if retrieve_token in self.synsets_info:
                concept_ids.append(
                    [self.cn_concept2id[synset['concept_name']] for synset in self.synsets_info[retrieve_token]])
                weights.append(
                    [synset['weight'] for synset in self.synsets_info[retrieve_token]])
            else:
                concept_ids.append([])
                weights.append([])
        return concept_ids,weights

    def __call__(self,
                 examples,
                 tokenizer,
                 max_seq_length,
                 max_question_length,
                 max_answer_length):
        for example in tqdm(examples):

            tokenization_info = self.all_tokenization_info[example.id]

            question_tokens = tokenizer.tokenize(example.question_text)
            assert question_tokens == tokenization_info['question_subtokens']
            if self.concept_settings['use_conceptnet']:
                question_cn_concepts, question_cn_weights = self._lookup_conceptnet_concept_ids(question_tokens,
                                                                                                tokenization_info[
                                                                                                    'question_sub_to_ori_index'],
                                                                                                tokenization_info[
                                                                                                    'question_tokens'],
                                                                                                tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                                                tokenizer=tokenizer)  # if tolower is True, tokenizer must be given

            if len(question_tokens) > max_question_length:
                question_tokens = question_tokens[0:max_question_length]
                question_cn_concepts = question_cn_concepts[0:max_question_length]
                question_cn_weights = question_cn_weights[0:max_question_length]
                
            answer_tokens = tokenizer.tokenize(example.answer_text)
            assert answer_tokens == tokenization_info['answer_subtokens']
            if self.concept_settings['use_conceptnet']:
                answer_cn_concepts, answer_cn_weights = self._lookup_conceptnet_concept_ids(answer_tokens,
                                                                                            tokenization_info[
                                                                                                'answer_sub_to_ori_index'],
                                                                                            tokenization_info[
                                                                                                'answer_tokens'],
                                                                                            tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                                            tokenizer=tokenizer)  # if tolower is True, tokenizer must be given

            if len(answer_tokens) > max_answer_length:
                answer_tokens = answer_tokens[0:max_answer_length]
                answer_cn_concepts = answer_cn_concepts[0:max_answer_length]
                answer_cn_weights = answer_cn_weights[0:max_answer_length]

            q_tokens = []
            q_segment_ids = []
            q_cn_concept_ids = []
            q_cn_concept_weights = []
    
            q_tokens.append("[CLS]")
            q_segment_ids.append(1)
            q_cn_concept_ids.append([])
            q_cn_concept_weights.append([])
            
            '''
            for token in  question_tokens:
                q_tokens.append(token)
                q_segment_ids.append(1)
            '''
            for token, question_cn_concept,question_cn_weight in zip(question_tokens, question_cn_concepts,question_cn_weights):
                q_tokens.append(token)
                q_segment_ids.append(0)
                q_cn_concept_ids.append(question_cn_concept)
                q_cn_concept_weights.append(question_cn_weight)
                
            q_tokens.append("[SEP]")
            q_segment_ids.append(1)
            q_cn_concept_ids.append([])
            q_cn_concept_weights.append([])
            
            a_tokens = []
            a_segment_ids = []
            a_cn_concept_ids = []
            a_cn_concept_weights = []
            
            a_tokens.append("[CLS]")
            a_segment_ids.append(1)
            a_cn_concept_ids.append([])
            a_cn_concept_weights.append([])
            
            '''
            for token in answer_tokens:
                a_tokens.append(token)
                a_segment_ids.append(1)
            '''
            for token, answer_cn_concept,answer_cn_weight in zip(answer_tokens, answer_cn_concepts,answer_cn_weights):
                a_tokens.append(token)
                a_segment_ids.append(1)
                a_cn_concept_ids.append(answer_cn_concept)
                a_cn_concept_weights.append(answer_cn_weight)

            a_tokens.append("[SEP]")
            a_segment_ids.append(1)
            a_cn_concept_ids.append([])
            a_cn_concept_weights.append([])

            q_input_mask = [1] * len(q_tokens)
            a_input_mask = [1] * len(a_tokens)

            while len(q_tokens) < max_question_length+2:
                q_tokens.append("[PAD]")
                q_segment_ids.append(1)
                q_input_mask.append(0)
                q_cn_concept_ids.append([])
                q_cn_concept_weights.append([])

            while len(a_tokens) < max_answer_length+2:
                a_tokens.append("[PAD]")
                a_segment_ids.append(1)
                a_input_mask.append(0)
                a_cn_concept_ids.append([])
                a_cn_concept_weights.append([])

            q_input_ids = tokenizer.convert_tokens_to_ids(q_tokens)
            a_input_ids = tokenizer.convert_tokens_to_ids(a_tokens)

            concept_ids, concept_weights, max_concept_length = q_cn_concept_ids, q_cn_concept_weights, self.max_cn_concept_length
            for cindex in range(len(concept_ids)):
                concept_ids[cindex] = concept_ids[cindex] + [0] * (max_concept_length - len(concept_ids[cindex]))
                concept_ids[cindex] = concept_ids[cindex][:max_concept_length]
                concept_weights[cindex] = concept_weights[cindex] + [0] * (
                            max_concept_length - len(concept_weights[cindex]))
                concept_weights[cindex] = concept_weights[cindex][:max_concept_length]
            assert all([len(id_list) == max_concept_length for id_list in concept_ids])
            assert all([len(id_list) == max_concept_length for id_list in concept_weights])
            q_cn_concept_ids = concept_ids
            q_cn_concept_weights = concept_weights

            concept_ids, concept_weights, max_concept_length = a_cn_concept_ids, a_cn_concept_weights, self.max_cn_concept_length
            for cindex in range(len(concept_ids)):
                concept_ids[cindex] = concept_ids[cindex] + [0] * (max_concept_length - len(concept_ids[cindex]))
                concept_ids[cindex] = concept_ids[cindex][:max_concept_length]
                concept_weights[cindex] = concept_weights[cindex] + [0] * (
                            max_concept_length - len(concept_weights[cindex]))
                concept_weights[cindex] = concept_weights[cindex][:max_concept_length]
            assert all([len(id_list) == max_concept_length for id_list in concept_ids])
            assert all([len(id_list) == max_concept_length for id_list in concept_weights])
            a_cn_concept_ids = concept_ids
            a_cn_concept_weights = concept_weights

            feature = InputFeatures(
                qas_id=example.id,
                q_tokens=q_tokens,
                q_input_ids=q_input_ids,
                q_input_mask=q_input_mask,
                q_segment_ids=q_segment_ids,
                q_pos_ids=range(max_question_length+2),
                q_cn_concept_ids=q_cn_concept_ids,
                q_cn_concept_weights=q_cn_concept_weights,
                a_tokens=a_tokens,
                a_input_ids=a_input_ids,
                a_input_mask=a_input_mask,
                a_segment_ids=a_segment_ids,
                a_pos_ids=range(max_answer_length+2),
                a_cn_concept_ids=a_cn_concept_ids,
                a_cn_concept_weights=a_cn_concept_weights,
                label=example.label)

            yield feature

def read_qas_examples(input_file):
    """Read a qas json file into a list of qasExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    examples = []
    for entry in input_data:
      id = entry["id"]
      question_text = entry["question"]
      answer_text = entry["answer"]
      label = entry["label"]
      example = QASExample(
          id = id,
          question_text=question_text,
          answer_text=answer_text,
          label = label)
      examples.append(example)
    return examples

class DataProcessor(object):
    def __init__(self,
                 args):
        self._bert_tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model)
        self.train_examples = None
        self.predict_examples = None
        self._max_seq_length = args.max_seq_length
        self._max_question_length = args.max_question_length
        self._max_answer_length = args.max_answer_length

        self.train_cn_max_concept_length = 200
        self.predict_cn_max_concept_length = None
        
    def get_examples(self, data_path):
        examples = read_qas_examples(input_file=data_path)
        return examples

    def get_features(self, examples):
        convert_examples_to_features = Examples_To_Features_Converter()
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self._bert_tokenizer,
            max_seq_length=self._max_seq_length,
            max_question_length=self._max_question_length,
            max_answer_length=self._max_answer_length)
        return features
    def data_generator(self, data_path, batch_size,index ,**concept_settings):
        #chunk_num = 0
        if phase == 'train':
            self.train_examples = self.get_examples(data_path)
            #chunk_num = int((len(self.train_examples) + split - 1) / split)
        elif phase == 'predict':
            self.predict_examples = self.get_examples(data_path)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")
        
        def batch_reader(features, batch_size):
            batch,q_input_ids,q_token_type_ids,q_attention_mask,q_position_ids,a_input_ids,a_token_type_ids,a_attention_mask,a_position_ids,label = [],[],[],[],[],[],[],[],[],[]
            q_cn_concept_ids, q_cn_concept_weights, a_cn_concept_ids, a_cn_concept_weights = [],[],[],[]
            batch_data = {}
            for (index, feature) in enumerate(features):
                q_input_ids.append(feature.q_input_ids)
                #q_token_type_ids.append(feature.q_segment_ids)
                q_attention_mask.append(feature.q_input_mask)
                #q_position_ids.append(feature.q_pos_ids)
                a_input_ids.append(feature.a_input_ids)
                #a_token_type_ids.append(feature.a_segment_ids)
                a_attention_mask.append(feature.a_input_mask)
                #a_position_ids.append(feature.a_pos_ids)
                label.append(feature.label)
                q_cn_concept_ids.append(feature.q_cn_concept_ids)
                q_cn_concept_weights.append(feature.q_cn_concept_weights)
                a_cn_concept_ids.append(feature.a_cn_concept_ids)
                a_cn_concept_weights.append(feature.a_cn_concept_weights)

                to_append = len(q_input_ids) < batch_size

                if to_append:
                    continue
                else:
                    batch_data["q_input_ids"] = q_input_ids
                    #batch_data["q_token_type_ids"] = q_token_type_ids
                    batch_data["q_attention_mask"] = q_attention_mask
                    #batch_data["q_position_ids"] = q_position_ids
                    batch_data["a_input_ids"] = a_input_ids
                    #batch_data["a_token_type_ids"] = a_token_type_ids
                    batch_data["a_attention_mask"] = a_attention_mask
                    #batch_data["a_position_ids"] = a_position_ids
                    batch_data["label"] = label
                    batch_data["q_cn_concept_ids"] = q_cn_concept_ids
                    batch_data["q_cn_concept_weights"] = q_cn_concept_weights
                    batch_data["a_cn_concept_ids"] = a_cn_concept_ids
                    batch_data["a_cn_concept_weights"] = a_cn_concept_weights
                    batch.append(batch_data)
                    batch_data = {}
                    q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, label = [], [], [], [], []
                    q_cn_concept_ids, q_cn_concept_weights, a_cn_concept_ids, a_cn_concept_weights = [], [], [], []

            return batch

        if phase == 'train':
            #self.train_examples = self.train_examples[index*chunk_num:min((index+1)*chunk_num,len(self.train_examples))]
            self.train_cn_max_concept_length = Examples_To_Features_Converter(**concept_settings).max_cn_concept_length
        else:
            self.predict_cn_max_concept_length = Examples_To_Features_Converter(
                **concept_settings).max_cn_concept_length

        if phase == 'train':
            features = self.get_features(self.train_examples)
        else:
            features = self.get_features(self.predict_examples)
        
        return batch_reader(features,batch_size)
        