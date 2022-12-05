import six
import math
import json
import random
import collections
import os
import pickle
import logging
import tokenization
from batching import prepare_batch_data

#from eval.squad_v1_official_evaluate import evaluate

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


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
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", answer_text: %s" % (
            tokenization.printable_text(self.answer_text))
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 concept_ids,
                 qas_id,
                 label,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.concept_ids = concept_ids
        self.label = label
        self.qas_id = qas_id

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
        self.concept2id = self.concept_settings['concept2id']

        # 3. retrieved related wordnet concepts (if use_wordnet)
        if concept_settings['use_wordnet']:
            assert not self.concept_settings['use_nell']
            retrieved_synset_filepath = self.concept_settings['retrieved_synset_path']
            assert os.path.exists(retrieved_synset_filepath)
            self.synsets_info = pickle.load(open(retrieved_synset_filepath, 'rb'))  # token to sysnet names
            self.max_concept_length = max([len(synsets) for synsets in self.synsets_info.values()])

        # 4. retrieved related nell concepts (if use_nell)
        if concept_settings['use_nell']:
            assert not self.concept_settings['use_wordnet']
            retrieved_nell_concept_filepath = self.concept_settings['retrieved_nell_concept_path']
            assert os.path.exists(retrieved_nell_concept_filepath)
            self.nell_retrieve_info = {}
            for item in pickle.load(open(retrieved_nell_concept_filepath, 'rb')):
                self.nell_retrieve_info[item['id']] = item
            self.max_concept_length = max([max([len(entity_info['retrieved_concepts']) for entity_info in
                                                item['question_entities'] + item['answer_entities']])
                                           for qid, item in self.nell_retrieve_info.items() if
                                           len(item['question_entities'] + item['answer_entities']) > 0])
        # return list of concept ids given input subword list

    def _lookup_wordnet_concept_ids(self, sub_tokens, sub_to_ori_index, tokens, tolower, tokenizer):
        concept_ids = []
        for index in range(len(sub_tokens)):
            original_token = tokens[sub_to_ori_index[index]]
            # if tokens are in upper case, we must lower it for retrieving
            retrieve_token = tokenizer.basic_tokenizer._run_strip_accents(
                original_token.lower()) if tolower else original_token
            if retrieve_token in self.synsets_info:
                concept_ids.append([self.concept2id[synset_name] for synset_name in self.synsets_info[retrieve_token]])
            else:
                concept_ids.append([])
        return concept_ids

    def _lookup_nell_concept_ids(self, sub_tokens, sub_to_ori_index, tokens, nell_info):
        original_concept_ids = [[] for _ in range(len(tokens))]
        for entity_info in nell_info:
            for pos in range(entity_info['token_start'], entity_info['token_end'] + 1):
                original_concept_ids[pos] += [self.concept2id[category_name] for category_name in
                                              entity_info['retrieved_concepts']]
        for pos in range(len(original_concept_ids)):
            original_concept_ids[pos] = list(set(original_concept_ids[pos]))
        concept_ids = [original_concept_ids[sub_to_ori_index[index]] for index in range(len(sub_tokens))]
        return concept_ids

    def __call__(self,
                 examples,
                 tokenizer,
                 max_seq_length,
                 doc_stride,
                 max_query_length,
                 is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        for (example_index, example) in enumerate(examples):
            tokenization_info = self.all_tokenization_info[example.id]
            question_tokens = tokenizer.tokenize(example.question_text)
            # check online subword tokenization result is the same as offline result
            assert question_tokens == tokenization_info['question_subtokens']
            if self.concept_settings['use_wordnet']:
                question_concepts = self._lookup_wordnet_concept_ids(question_tokens,
                                                                  tokenization_info['question_sub_to_ori_index'],
                                                                  tokenization_info['question_tokens'],
                                                                  tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                  tokenizer=tokenizer)  # if tolower is True, tokenizer must be given

            if self.concept_settings['use_nell']:
                question_concepts = self._lookup_nell_concept_ids(question_tokens,
                                                               tokenization_info['question_sub_to_ori_index'],
                                                               tokenization_info['question_tokens'],
                                                               self.nell_retrieve_info[example.id][
                                                                   'question_entities'])

            if len(question_tokens) > max_query_length:
                question_tokens = question_tokens[0:max_query_length]
                question_concepts = question_concepts[0:max_query_length]

            answer_tokens = tokenizer.tokenize(example.answer_text)
            # check online subword tokenization result is the same as offline result
            print(answer_tokens)
            print(tokenization_info['answer_subtokens'])
            assert answer_tokens == tokenization_info['answer_subtokens']
            if self.concept_settings['use_wordnet']:
                answer_concepts = self._lookup_wordnet_concept_ids(answer_tokens,
                                                                     tokenization_info['answer_sub_to_ori_index'],
                                                                     tokenization_info['answer_tokens'],
                                                                     tolower=tokenizer.basic_tokenizer.do_lower_case == False,
                                                                     tokenizer=tokenizer)  # if tolower is True, tokenizer must be given

            if self.concept_settings['use_nell']:
                answer_concepts = self._lookup_nell_concept_ids(answer_tokens,
                                                                  tokenization_info['answer_sub_to_ori_index'],
                                                                  tokenization_info['answer_tokens'],
                                                                  self.nell_retrieve_info[example.id][
                                                                      'answer_entities'])

            if len(answer_tokens) > max_query_length:
                answer_tokens = answer_tokens[0:max_query_length]
                answer_concepts = answer_concepts[0:max_query_length]

            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            concept_ids = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            concept_ids.append([])
            for token, query_concept in zip(question_tokens, question_concepts):
                tokens.append(token)
                segment_ids.append(0)
                concept_ids.append(query_concept)
            tokens.append("[SEP]")
            segment_ids.append(0)
            concept_ids.append([])

            for token, query_concept in zip(answer_tokens, answer_concepts):
                tokens.append(token)
                segment_ids.append(1)
                concept_ids.append(query_concept)
            tokens.append("[SEP]")
            segment_ids.append(1)
            concept_ids.append([])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            for cindex in range(len(concept_ids)):
                concept_ids[cindex] = concept_ids[cindex] + [0] * (self.max_concept_length - len(concept_ids[cindex]))
                concept_ids[cindex] = concept_ids[cindex][:self.max_concept_length]
            assert all([len(id_list) == self.max_concept_length for id_list in concept_ids])

            label = None
            if is_training:
                label = example.label

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=None,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                concept_ids=concept_ids,
                qas_id=example.id,
                label=label,
                start_position=None,
                end_position=None,
                is_impossible=None)

            unique_id += 1

            yield feature

def read_qas_examples(input_file, is_training):
    """Read a qas json file into a list of qasExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    '''
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    '''
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
    def __init__(self, vocab_path, do_lower_case, max_seq_length, in_tokens,
                 doc_stride, max_query_length):
        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self._max_seq_length = max_seq_length
        self._doc_stride = doc_stride
        self._max_query_length = max_query_length
        self._in_tokens = in_tokens

        self.vocab = self._tokenizer.vocab
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

        self.current_train_example = -1
        self.num_train_examples = -1
        self.current_train_epoch = -1

        self.train_examples = None
        self.predict_examples = None
        self.num_examples = {'train': -1, 'predict': -1}

        self.train_max_concept_length = None
        self.predict_max_concept_length = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_train_example, self.current_train_epoch

    def get_examples(self,
                     data_path,
                     is_training
                     ):
                     #version_2_with_negative=False):
        examples = read_qas_examples(
            input_file=data_path,
            is_training=is_training)
            #version_2_with_negative=version_2_with_negative)
        return examples

    def get_num_examples(self, phase):
        if phase not in ['train', 'predict']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")
        return self.num_examples[phase]

    def get_features(self, examples, is_training, **concept_settings):
        convert_examples_to_features = Examples_To_Features_Converter(**concept_settings)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=self._tokenizer,
            max_seq_length=self._max_seq_length,
            doc_stride=self._doc_stride,
            max_query_length=self._max_query_length,
            is_training=is_training)
        return features

    def data_generator(self,
                       data_path,
                       batch_size,
                       phase='train',
                       shuffle=False,
                       dev_count=1,
                       version_2_with_negative=False,
                       epoch=1,
                       **concept_settings):
        if phase == 'train':
            self.train_examples = self.get_examples(
                data_path,
                is_training=True
            )
                #无效参数
                #version_2_with_negative=version_2_with_negative)
            examples = self.train_examples
            self.num_examples['train'] = len(self.train_examples)
        elif phase == 'predict':
            self.predict_examples = self.get_examples(
                data_path,
                is_training=False
            )
                #无效参数
                #version_2_with_negative=version_2_with_negative)
            examples = self.predict_examples
            self.num_examples['predict'] = len(self.predict_examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'predict'].")

        def batch_reader(features, batch_size, in_tokens):
            batch, total_token_num, max_len = [], 0, 0
            for (index, feature) in enumerate(features):
                if phase == 'train':
                    self.current_train_example = index + 1
                seq_len = len(feature.input_ids)
                label = [feature.qas_id, feature.unique_id] if feature.label is None else [feature.label]
                example = [
                               #feature.input_ids, feature.segment_ids, range(seq_len), feature.concept_ids
                              feature.input_ids, feature.segment_ids, range(200), feature.concept_ids
                          ] + label
                max_len = max(max_len, seq_len)

                # max_len = max(max_len, len(token_ids))
                if in_tokens:
                    to_append = (len(batch) + 1) * max_len <= batch_size
                else:
                    to_append = len(batch) < batch_size

                if to_append:
                    batch.append(example)
                    total_token_num += seq_len
                else:
                    yield batch, total_token_num
                    batch, total_token_num, max_len = [example
                                                       ], seq_len, seq_len
            if len(batch) > 0:
                yield batch, total_token_num

        if phase == 'train':
            self.train_max_concept_length = Examples_To_Features_Converter(**concept_settings).max_concept_length
        else:
            self.predict_max_concept_length = Examples_To_Features_Converter(**concept_settings).max_concept_length

        def wrapper():
            for epoch_index in range(epoch):
                if shuffle:
                    random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                    features = self.get_features(examples, is_training=True, **concept_settings)
                    max_concept_length = self.train_max_concept_length
                else:
                    features = self.get_features(examples, is_training=False, **concept_settings)
                    max_concept_length = self.predict_max_concept_length

                all_dev_batches = []
                for batch_data, total_token_num in batch_reader(
                        features, batch_size, self._in_tokens):
                    batch_data = prepare_batch_data(
                        batch_data,
                        total_token_num,
                        voc_size=-1,
                        pad_id=self.pad_id,
                        cls_id=self.cls_id,
                        sep_id=self.sep_id,
                        mask_id=-1,
                        return_input_mask=True,
                        return_max_len=False,
                        return_num_token=False,
                        max_concept_length=max_concept_length)
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)

                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      version_2_with_negative, null_score_diff_threshold,
                      verbose, predict_file, evaluation_result_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))
    logger.info("Writing evaluation result to: %s" % (evaluation_result_file))

    # load ground truth file for evaluation and post-edit
    with open(predict_file, "r", encoding='utf-8') as reader:
        predict_json = json.load(reader)["data"]

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[
                    0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1
                                                              )]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case,
                                            verbose)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        # debug
        if best_non_null_entry is None:
            logger.info("Emmm..., sth wrong")

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    eval_result = evaluate(predict_json, all_predictions)

    with open(evaluation_result_file, "w") as writer:
        writer.write(json.dumps(eval_result, indent=4) + "\n")

    return eval_result


def get_final_text(pred_text, orig_text, do_lower_case, verbose):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return