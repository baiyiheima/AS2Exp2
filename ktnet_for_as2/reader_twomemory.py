import json
import os
import pickle
from transformers import BertModel, BertTokenizer
from data_preprocessing import tokenization


class QASExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
            id,
            question_text,
            #question_entities_strset,
            answer_text,
            #answer_entities_strset,
            label
            ):
      self.id = id
      self.question_text = question_text
      #self.question_entities_strset = question_entities_strset
      self.answer_text = answer_text
      #self.answer_entities_strset = answer_entities_strset
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
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 wn_concept_ids,
                 nell_concept_ids,
                 label):
        self.qas_id = qas_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.wn_concept_ids = wn_concept_ids
        self.nell_concept_ids = nell_concept_ids
        self.label = label


class Examples_To_Features_Converter(object):
    def __init__(self, **concept_settings):
        self.concept_settings = concept_settings

        # load necessary data files for mapping to related concepts
        # 1. mapping from subword-level tokenization to word-level tokenization
        tokenization_filepath = self.concept_settings['tokenization_path']
        assert os.path.exists(tokenization_filepath)
        self.all_tokenization_info = {}
        for item in pickle.load(open(tokenization_filepath, 'rb')):
            self.all_tokenization_info[item['id']] = item

        # 2. mapping from concept name to concept id (currently only support one KB)
        self.wn_concept2id = self.concept_settings['wn_concept2id']
        self.nell_concept2id = self.concept_settings['nell_concept2id']

        # 3. retrieved related wordnet concepts (if use_wordnet)
        if concept_settings['use_wordnet']:
            retrieved_synset_filepath = self.concept_settings['retrieved_synset_path']
            assert os.path.exists(retrieved_synset_filepath)
            self.synsets_info = pickle.load(open(retrieved_synset_filepath, 'rb'))  # token to sysnet names
            self.max_wn_concept_length = max([len(synsets) for synsets in self.synsets_info.values()])
            self.max_wn_concept_length = max([len(synsets) for synsets in self.synsets_info.values()])

        # 4. retrieved related nell concepts (if use_nell)
        if concept_settings['use_nell']:
            retrieved_nell_concept_filepath = self.concept_settings['retrieved_nell_concept_path']
            assert os.path.exists(retrieved_nell_concept_filepath)
            self.nell_retrieve_info = {}
            for item in pickle.load(open(retrieved_nell_concept_filepath, 'rb')):
                self.nell_retrieve_info[item['id']] = item
            self.max_nell_concept_length = max([max([len(entity_info['retrieved_concepts']) for entity_info in
                                                     item['question_entities'] + item['answer_entities']])
                                                for qid, item in self.nell_retrieve_info.items() if
                                                len(item['question_entities'] + item['answer_entities']) > 0])

    def _lookup_wordnet_concept_ids(self, sub_tokens, sub_to_ori_index, tokens, tolower, tokenizer):
        concept_ids = []
        for index in range(len(sub_tokens)):
            original_token = tokens[sub_to_ori_index[index]]
            # if tokens are in upper case, we must lower it for retrieving
            retrieve_token = tokenizer.basic_tokenizer._run_strip_accents(
                original_token.lower()) if tolower else original_token
            if retrieve_token in self.synsets_info:
                concept_ids.append(
                    [self.wn_concept2id[synset_name] for synset_name in self.synsets_info[retrieve_token]])
            else:
                concept_ids.append([])
        return concept_ids

    def _lookup_nell_concept_ids(self, sub_tokens, sub_to_ori_index, tokens, nell_info):
        original_concept_ids = [[] for _ in range(len(tokens))]
        for entity_info in nell_info:
            for pos in range(entity_info['token_start'], entity_info['token_end'] + 1):
                original_concept_ids[pos] += [self.nell_concept2id[category_name] for category_name in
                                              entity_info['retrieved_concepts']]
        for pos in range(len(original_concept_ids)):
            original_concept_ids[pos] = list(set(original_concept_ids[pos]))
        concept_ids = [original_concept_ids[sub_to_ori_index[index]] for index in range(len(sub_tokens))]
        return concept_ids

    def __call__(self,
                 examples,
                 tokenizer,
                 max_seq_length,
                 max_question_length,
                 max_answer_length):
        for(example_idx, example) in enumerate(examples):

            tokenization_info = self.all_tokenization_info[example.id]

            question_tokens = tokenizer.tokenize(example.question_text)
            assert question_tokens == tokenization_info['question_subtokens']
            if self.concept_settings['use_wordnet']:
                question_wn_concepts = self._lookup_wordnet_concept_ids(question_tokens,
                                                                     tokenization_info['question_sub_to_ori_index'],
                                                                     tokenization_info['question_tokens'],
                                                                     tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                     tokenizer=tokenizer)  # if tolower is True, tokenizer must be given
            if self.concept_settings['use_nell']:
                question_nell_concepts = self._lookup_nell_concept_ids(question_tokens,
                                                                    tokenization_info['question_sub_to_ori_index'],
                                                                    tokenization_info['question_tokens'],
                                                                    self.nell_retrieve_info[example.id][
                                                                        'question_entities'])
            if len(question_tokens) > max_question_length:
                question_tokens = question_tokens[0:max_question_length]
                question_wn_concepts = question_wn_concepts[0:max_question_length]
                question_nell_concepts = question_nell_concepts[0:max_question_length]

            answer_tokens = tokenizer.tokenize(example.answer_text)
            assert answer_tokens == tokenization_info['answer_subtokens']
            if self.concept_settings['use_wordnet']:
                answer_wn_concepts = self._lookup_wordnet_concept_ids(answer_tokens,
                                                                        tokenization_info['answer_sub_to_ori_index'],
                                                                        tokenization_info['answer_tokens'],
                                                                        tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                        tokenizer=tokenizer)  # if tolower is True, tokenizer must be given
            if self.concept_settings['use_nell']:
                answer_nell_concepts = self._lookup_nell_concept_ids(answer_tokens,
                                                                       tokenization_info['answer_sub_to_ori_index'],
                                                                       tokenization_info['answer_tokens'],
                                                                       self.nell_retrieve_info[example.id][
                                                                           'answer_entities'])
            if len(answer_tokens) > max_answer_length:
                answer_tokens = answer_tokens[0:max_answer_length]
                answer_wn_concepts = answer_wn_concepts[0:max_answer_length]
                answer_nell_concepts = answer_nell_concepts[0:max_answer_length]

            tokens = []
            segment_ids = []
            wn_concept_ids = []
            nell_concept_ids = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            wn_concept_ids.append([])
            nell_concept_ids.append([])

            for token, question_wn_concept, question_nell_concept in  zip(question_tokens, question_wn_concepts, question_nell_concepts):
                tokens.append(token)
                segment_ids.append(0)
                wn_concept_ids.append(question_wn_concept)
                nell_concept_ids.append(question_nell_concept)

            tokens.append("[SEP]")
            segment_ids.append(0)
            wn_concept_ids.append([])
            nell_concept_ids.append([])

            for token, answer_wn_concept, answer_nell_concept in zip(answer_tokens, answer_wn_concepts,
                                                                         answer_nell_concepts):
                tokens.append(token)
                segment_ids.append(1)
                wn_concept_ids.append(answer_wn_concept)
                nell_concept_ids.append(answer_nell_concept)

            tokens.append("[SEP]")
            segment_ids.append(1)
            wn_concept_ids.append([])
            nell_concept_ids.append([])

            input_mask = [1] * len(tokens)

            while len(tokens) < max_seq_length:
                tokens.append("[PAD]")
                segment_ids.append(1)
                wn_concept_ids.append([])
                nell_concept_ids.append([])
                input_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            for concept_ids, max_concept_length in zip((wn_concept_ids, nell_concept_ids),
                                                       (self.max_wn_concept_length, self.max_nell_concept_length)):
                for cindex in range(len(concept_ids)):
                    concept_ids[cindex] = concept_ids[cindex] + [0] * (max_concept_length - len(concept_ids[cindex]))
                    concept_ids[cindex] = concept_ids[cindex][:max_concept_length]
                assert all([len(id_list) == max_concept_length for id_list in concept_ids])

            feature = InputFeatures(
                qas_id=example.id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                wn_concept_ids=wn_concept_ids,
                nell_concept_ids=nell_concept_ids,
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
      #question_entities_strset = set([entity_info["text"] for entity_info in entry["question_entities"]])
      answer_text = entry["answer"]
      #answer_entities_strset = set([entity_info["text"] for entity_info in entry["answer_entities"]])
      label = entry["label"]
      #print(type(question_text))
      example = QASExample(
          id = id,
          question_text=question_text,
          #question_entities_strset=question_entities_strset,
          answer_text=answer_text,
          #answer_entities_strset=answer_entities_strset,
          label = label)
      #print(type(example.question_text))
      examples.append(example)
    return examples

class DataProcessor(object):
    def __init__(self,
                 do_lower_case,
                 max_seq_length,
                 max_question_length,
                 max_answer_length):
        self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' if do_lower_case else 'bert-base-cased')
        self.train_examples = None
        self.predict_examples = None
        self._max_seq_length = max_seq_length
        self._max_question_length = max_question_length
        self._max_answer_length = max_answer_length

        self.train_wn_max_concept_length = None
        self.predict_wn_max_concept_length = None
        self.train_nell_max_concept_length = None
        self.predict_nell_max_concept_length = None

    def get_examples(self, data_path):
        examples = read_qas_examples(input_file=data_path)
        return examples

    def get_features(self, examples, **concept_settings):
        convert_examples_to_features = Examples_To_Features_Converter(**concept_settings)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self._bert_tokenizer,
            max_seq_length=self._max_seq_length,
            max_question_length=self._max_question_length,
            max_answer_length=self._max_answer_length)
        return features
    def data_generator(self, data_path, batch_size, phase, **concept_settings):

        if phase == 'train':
            self.train_examples = self.get_examples(data_path)
        elif phase == 'predict':
            self.predict_examples = self.get_examples(data_path)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")
        
        def batch_reader(features, batch_size):
            batch,input_ids,token_type_ids,attention_mask,wn_concept_ids,nell_concept_ids,label = [],[],[],[],[],[],[]
            batch_data = {}
            for (index, feature) in enumerate(features):
                input_ids.append(feature.input_ids)
                token_type_ids.append(feature.segment_ids)
                attention_mask.append(feature.input_mask)
                wn_concept_ids.append(feature.wn_concept_ids)
                nell_concept_ids.append(feature.nell_concept_ids)
                label.append(feature.label)

                to_append = len(input_ids) < batch_size

                if to_append:
                    continue
                else:
                    batch_data["input_ids"] = input_ids
                    batch_data["token_type_ids"] = token_type_ids
                    batch_data["attention_mask"] = attention_mask
                    batch_data["wn_concept_ids"] = wn_concept_ids
                    batch_data["nell_concept_ids"] = nell_concept_ids
                    batch_data["label"] = label
                    batch.append(batch_data)
                    batch_data = {}
                    input_ids, token_type_ids, attention_mask, wn_concept_ids, nell_concept_ids, label = [], [], [], [], [], []
            return batch

        if phase == 'train':
            self.train_wn_max_concept_length = Examples_To_Features_Converter(**concept_settings).max_wn_concept_length
            self.train_nell_max_concept_length = Examples_To_Features_Converter(
                **concept_settings).max_nell_concept_length
        else:
            self.predict_wn_max_concept_length = Examples_To_Features_Converter(
                **concept_settings).max_wn_concept_length
            self.predict_nell_max_concept_length = Examples_To_Features_Converter(
                **concept_settings).max_nell_concept_length

        if phase == 'train':
            features = self.get_features(self.train_examples, **concept_settings)
        else:
            features = self.get_features(self.predict_examples, **concept_settings)
        
        return batch_reader(features,batch_size)
        