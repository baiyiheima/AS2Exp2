#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import multiprocessing
import os
import time
import logging
import random
import numpy as np
import paddle.fluid as fluid

from imap_qa import calc_map1, calc_mrr1
from reader.wikiqa import DataProcessor, write_predictions
from model.bert import BertConfig, BertModel
from model.layers import MemoryLayer, TriLinearTwoTimeSelfAttentionLayer
from utils.args import ArgumentGroup, print_arguments
from optimization import optimization
from utils.init import init_pretraining_params, init_checkpoint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# yapf: disable
parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path", str, "../cased_L-12_H-768_A-12/bert_config.json", "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, "../cased_L-12_H-768_A-12/params",
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "./output", "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps", int, 4000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000,
                "The steps interval for validation (effective only when do_val is True).")
train_g.add_arg("use_ema", bool, True, "Whether to use ema.")
train_g.add_arg("ema_decay", float, 0.9999, "Decay rate for expoential moving average.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file", str, "../data/WikiQA/wikicTrain.json", "SQuAD json for training. E.g., train-v1.1.json.")
data_g.add_arg("predict_file", str, "../data/WikiQA/wikicTest.json", "SQuAD json for predictions. E.g. dev-v1.1.json or test-v1.1.json.")
data_g.add_arg("raw_predict_file", str, "../data/WikiQA/wikic_test.tsv", ".")
data_g.add_arg("vocab_path", str, "../cased_L-12_H-768_A-12/vocab.txt", "Vocabulary path.")
data_g.add_arg("version_2_with_negative", bool, False,
               "If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, it should be set true.")
data_g.add_arg("max_seq_len", int, 200, "Number of words of the longest seqence.")
data_g.add_arg("max_query_length", int, 100, "Max query length.")
#data_g.add_arg("max_answer_length", int, 30, "Max answer length.")
data_g.add_arg("batch_size", int, 12, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, False,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("doc_stride", int, 128,
               "When splitting up a long document into chunks, how much stride to take between chunks.")
data_g.add_arg("n_best_size", int, 20,
               "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
data_g.add_arg("null_score_diff_threshold", float, 0.0,
               "If null_score - best_non_null is greater than the threshold predict null.")
data_g.add_arg("random_seed", int, 45, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 1, "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform validation during training.")
run_type_g.add_arg("do_predict", bool, True, "Whether to perform prediction.")
run_type_g.add_arg("freeze", bool, False, "freeze bert parameters")

mem_settings_g = ArgumentGroup(parser, "memory", "memory settings.")
mem_settings_g.add_arg('concept_embedding_path', str, "../retrieve_concepts/KB_embeddings/wn_concept2vec.txt", 'path of pretrained concept file')
mem_settings_g.add_arg('use_wordnet', bool, True, 'whether to use wordnet memory')
mem_settings_g.add_arg('retrieved_synset_path', str,
                       '../retrieve_concepts/retrieve_wordnet/output_wikiqa/retrived_synsets.data',
                       'path of retrieved synsets')
mem_settings_g.add_arg('use_nell', bool, False, 'whether to use nell memory')
mem_settings_g.add_arg('train_retrieved_nell_concept_path', str,
                       '../retrieve_concepts/retrieve_nell/output_wikiqa/train.retrieved_nell_concepts.data',
                       'path of retrieved concepts for trainset')
mem_settings_g.add_arg('dev_retrieved_nell_concept_path', str,
                       '../retrieve_concepts/retrieve_nell/output_wikiqa/test.retrieved_nell_concepts.data',
                       'path of retrieved concepts for devset')

args = parser.parse_args()


# yapf: enable.

def create_model(pyreader_name, bert_config, max_concept_length, concept_embedding_mat, is_training=False,
                 freeze=False):
    if is_training:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, max_concept_length, 1],
                    [-1, args.max_seq_len, 1], [-1, 1]],
            dtypes=[
                'int64', 'int64', 'int64', 'int64', 'float32', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)
        (src_ids, pos_ids, sent_ids, concept_ids, input_mask, label) = fluid.layers.read_file(pyreader)
    else:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, max_concept_length, 1],
                    [-1, args.max_seq_len, 1], [-1, 1],  [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'int64', 'float32', 'int64','int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)
        (src_ids, pos_ids, sent_ids, concept_ids, input_mask, qas_id, unique_id) = fluid.layers.read_file(pyreader)

    '''1st Layer: BERT Layer'''
    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_config,
        use_fp16=args.use_fp16)

    enc_out = bert.get_sequence_output()
    if freeze:
        enc_out.stop_gradient = True
    logger.info("enc_out.stop_gradient: {}".format(enc_out.stop_gradient))

    '''
    #2nd layer: Memory Layer
    # get memory embedding
    concept_vocab_size = concept_embedding_mat.shape[0]
    concept_dim = concept_embedding_mat.shape[1]
    #(-1,384,49,100)
    memory_embs = fluid.layers.embedding(concept_ids,
                                         size=(concept_vocab_size, concept_dim),
                                         param_attr=fluid.ParamAttr(name="concept_emb_mat",
                                                                    do_model_average=False,
                                                                    trainable=False),
                                         dtype='float32')

    # get memory length
    concept_ids_reduced = fluid.layers.equal(concept_ids,
                                             fluid.layers.fill_constant(shape=[1], value=0,
                                                                        dtype="int64"))  # [batch_size, sent_size, concept_size, 1]
    concept_ids_reduced = fluid.layers.cast(concept_ids_reduced,
                                            dtype="float32")  # [batch_size, sent_size, concept_size, 1]
    concept_ids_reduced = fluid.layers.scale(
        fluid.layers.elementwise_sub(
            concept_ids_reduced,
            fluid.layers.fill_constant([1], "float32", 1)
        ),
        scale=-1
    )
    mem_length = fluid.layers.reduce_sum(concept_ids_reduced, dim=2)  # [batch_size, sent_size, 1]    

    # select and integrate
    memory_layer = MemoryLayer(bert_config, max_concept_length, concept_dim, mem_method='cat')
    memory_output = memory_layer.forward(enc_out, memory_embs, mem_length, ignore_no_memory_token=True)

    #3rd layer: Self-Matching Layer
    # calculate input dim for self-matching layer
    if memory_layer.mem_method == 'add':
        memory_output_size = bert_config['hidden_size']
    elif memory_layer.mem_method == 'cat':
        memory_output_size = bert_config['hidden_size'] + concept_dim
    else:
        raise ValueError("memory_layer.mem_method must be 'add' or 'cat'")
    logger.info("memory_output_size: {}".format(memory_output_size))

    # do matching
    self_att_layer = TriLinearTwoTimeSelfAttentionLayer(
        memory_output_size, dropout_rate=0.0,
        cat_mul=True, cat_sub=True, cat_twotime=True,
        cat_twotime_mul=False, cat_twotime_sub=True)  # [bs, sq, concat_hs]
    att_output = self_att_layer.forward(memory_output, input_mask)  # [bs, sq, concat_hs]
    att_output = fluid.layers.unstack(att_output, 1)
    cls_output = att_output[0]
    '''

    '''4th layer: Output Layer'''
    enc_out = fluid.layers.unstack(enc_out, 1)
    cls_output = enc_out[0]
    logits = fluid.layers.fc(
        input=cls_output,
        size=2,
        num_flatten_dims=1,
        param_attr=fluid.ParamAttr(
            name="cls_squad_out_w",
            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=bert_config['initializer_range'])),
        bias_attr=fluid.ParamAttr(
            name="cls_squad_out_b", initializer=fluid.initializer.Constant(0.)),
        act='softmax')

    #logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    #label_logits = fluid.layers.unstack(x=logits, axis=0)
    #label_logits = fluid.layers.squeeze(input=logits,axes=[0])
    label_logits = logits

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=label_logits, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    if is_training:

        def compute_loss(logits, positions):
            loss = fluid.layers.cross_entropy(
                input=logits, label=positions)
            loss = fluid.layers.mean(x=loss)
            return loss
        '''
        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0
        '''
        total_loss = compute_loss(label_logits,label)
        if args.use_fp16 and args.loss_scaling > 1.0:
            total_loss = total_loss * args.loss_scaling

        return pyreader, total_loss, num_seqs, label
    else:
        return pyreader, qas_id, unique_id, label_logits, num_seqs


RawResult = collections.namedtuple("RawResult",
                                   ["qas_id", "unique_id", "label_logits"])


def read_data(filename):
        with open(filename, 'r') as datafile:
            res = []
            count = 0
            for line in datafile:
                count = count + 1
                if (count == 1):
                    continue
                line = line.strip().split('\t')
                lines = []
                length = len(line)
                if (length < 3):
                    lines.append(line[0])
                    lines.append("question")
                    lines.append(line[1])
                else:
                    lines.append(line[0])
                    lines.append(line[1])
                    lines.append(line[2])

                res.append([lines[0], lines[1], float(lines[2])])
        return res

def predict(test_exe, test_program, test_pyreader, fetch_list, processor, eval_concept_settings,
            steps):
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    test_pyreader.start()
    all_results = []
    time_begin = time.time()
    all_preds = []
    while True:
        try:
            np_qas_ids, np_unique_ids, np_label_logits,  np_num_seqs = test_exe.run(
                fetch_list=fetch_list, program=test_program)
            for idx in range(np_unique_ids.shape[0]):
                if len(all_results) % 1000 == 0:
                    logger.info("Processing example: %d" % len(all_results))
                qas_id = int(np_qas_ids[idx])
                unique_id = int(np_unique_ids[idx])
                label_logits = [float(x) for x in np_label_logits[idx].flat]
                #end_logits = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(
                        qas_id = qas_id,
                        unique_id=unique_id,
                        label_logits=label_logits))
                all_preds.append(label_logits[1])
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    t_f = read_data(args.raw_predict_file)
    map = calc_map1(t_f, all_preds)
    mrr = calc_mrr1(t_f, all_preds)
    with open('output/result.txt', mode='a+', encoding='utf-8') as file_obj:
        if steps != None :
            file_obj.write("Steps:{} Eval performance:\n* MAP: {}\n* MRR: {}\n".format(steps, map, mrr))
        else:
            file_obj.write("Final Eval performance:\n* MAP: {}\n* MRR: {}\n".format(map, mrr))

    return map, mrr

    '''
    features = processor.get_features(
        processor.predict_examples, is_training=False, **eval_concept_settings)
    
    eval_result = write_predictions(processor.predict_examples, features, all_results,
                                    args.n_best_size, args.max_answer_length,
                                    args.do_lower_case, output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file,
                                    args.version_2_with_negative,
                                    args.null_score_diff_threshold, args.verbose, args.predict_file,
                                    output_evaluation_result_file)
    return eval_result
    '''

def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return id2concept, concept2id, embedding_mat


def train(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if not (args.do_train or args.do_predict or args.do_val):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    id2concept, concept2id, concept_embedding_mat = read_concept_embedding(
        args.concept_embedding_path)

    processor = DataProcessor(
        vocab_path=args.vocab_path,
        do_lower_case=args.do_lower_case,
        max_seq_length=args.max_seq_len,
        in_tokens=args.in_tokens,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length)

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    if args.do_train:
        train_concept_settings = {
            'tokenization_path': '../retrieve_concepts/tokenization_wikiqa/tokens/wikiqa.train.tokenization.{}.data'.format(
                'uncased' if args.do_lower_case else 'cased'),
            'concept2id': concept2id,
            'use_wordnet': args.use_wordnet,
            'retrieved_synset_path': args.retrieved_synset_path,
            'use_nell': args.use_nell,
            'retrieved_nell_concept_path': args.train_retrieved_nell_concept_path,
        }
        train_data_generator = processor.data_generator(
            data_path=args.train_file,
            batch_size=args.batch_size,
            phase='train',
            shuffle=True,
            dev_count=dev_count,
            version_2_with_negative=args.version_2_with_negative,
            epoch=args.epoch,
            **train_concept_settings)

        num_train_examples = processor.get_num_examples(phase='train')
        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size) // dev_count
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        logger.info("Device count: %d" % dev_count)
        logger.info("Num train examples: %d" % num_train_examples)
        logger.info("Max train steps: %d" % max_train_steps)
        logger.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()
        # if args.random_seed is not None:
        #     train_program.random_seed = args.random_seed
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, num_seqs,label = create_model(
                    pyreader_name='train_reader',
                    bert_config=bert_config,
                    max_concept_length=processor.train_max_concept_length,
                    concept_embedding_mat=concept_embedding_mat,
                    is_training=True,
                    freeze=args.freeze)

                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)

                if args.use_ema:
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                    ema.update()

                #从1.6版本开始此接口不再推荐使用，请不要在新写的代码中使用它，1.6+版本已默认开启更优的存储优化策略
                #fluid.memory_optimize(train_program, skip_opt_set=[loss.name, num_seqs.name])

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            logger.info("Theoretical memory usage in training:  %.3f - %.3f %s" %
                        (lower_mem, upper_mem, unit))

    if args.do_predict or args.do_val:
        eval_concept_settings = {
            'tokenization_path': '../retrieve_concepts/tokenization_wikiqa/tokens/wikiqa.test.tokenization.{}.data'.format(
                'uncased' if args.do_lower_case else 'cased'),
            'concept2id': concept2id,
            'use_wordnet': args.use_wordnet,
            'retrieved_synset_path': args.retrieved_synset_path,
            'use_nell': args.use_nell,
            'retrieved_nell_concept_path': args.dev_retrieved_nell_concept_path,
        }
        eval_data_generator = processor.data_generator(
            data_path=args.predict_file,
            batch_size=args.batch_size,
            phase='predict',
            shuffle=False,
            dev_count=1,
            epoch=1,
            **eval_concept_settings)

        test_prog = fluid.Program()
        # if args.random_seed is not None:
        #     test_prog.random_seed = args.random_seed
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, qas_id, unique_ids, label_logits, num_seqs = create_model(
                    pyreader_name='test_reader',
                    bert_config=bert_config,
                    max_concept_length=processor.predict_max_concept_length,
                    concept_embedding_mat=concept_embedding_mat,
                    is_training=False)

                if args.use_ema and 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)

                #fluid.memory_optimize(test_prog, skip_opt_set=[unique_ids.name, start_logits.name, end_logits.name, num_seqs.name])

        test_prog = test_prog.clone(for_test=True)
        # if args.random_seed is not None:
        #     test_prog.random_seed = args.random_seed

    exe.run(startup_prog)

    if args.do_train:
        logger.info('load pretrained concept embedding')
        #fluid.global_scope().find_var('concept_emb_mat').get_tensor().set(concept_embedding_mat, place)

        if args.init_checkpoint and args.init_pretraining_params:
            logger.info(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_predict or args.do_val:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing prediction!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)

        train_pyreader.decorate_tensor_provider(train_data_generator)

        train_pyreader.start()
        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()

        #while steps < 0 :
        while steps < max_train_steps:
            try:
                steps += 1
                print(steps)
                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        fetch_list = [label.name, loss.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, scheduled_lr.name, num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_num_seqs = outputs
                    else:
                        np_loss, np_lr, np_num_seqs = outputs
                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        logger.info(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, epoch = processor.get_train_progress()

                    logger.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                                "speed: %f steps/s" %
                                (epoch, current_example, num_train_examples, steps,
                                 np.sum(total_cost) / np.sum(total_num_seqs),
                                 args.skip_steps / used_time))
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()
                '''
                if steps % args.save_steps == 0 or steps == max_train_steps:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                '''
                if steps % args.validation_steps == 0 or steps == max_train_steps:
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator(
                                data_path=args.predict_file,
                                batch_size=args.batch_size,
                                phase='predict',
                                shuffle=False,
                                dev_count=1,
                                epoch=1,
                                **eval_concept_settings)
                        )
                        map, mrr = predict(exe, test_prog, test_pyreader, [
                            qas_id.name, unique_ids.name, label_logits.name , num_seqs.name
                        ], processor, eval_concept_settings, steps)

                        logger.info("Validation performance after step {}:\n* map: {}\n* mrr: {}".format(steps, map, mrr))


            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    if args.do_predict:
        test_pyreader.decorate_tensor_provider(eval_data_generator)

        if args.use_ema:
            with ema.apply(exe):
                map,mrr = predict(exe, test_prog, test_pyreader, [
                    qas_id.name, unique_ids.name, label_logits.name, num_seqs.name
                ], processor, eval_concept_settings,None)
        else:
            map,mrr = predict(exe, test_prog, test_pyreader, [
                qas_id.name, unique_ids.name, label_logits.name, num_seqs.name
            ], processor, eval_concept_settings,None)

        logger.info("Eval performance:\n* MAP: {}\n* MRR: {}".format(map,mrr))



if __name__ == '__main__':
    print_arguments(args)
    train(args)