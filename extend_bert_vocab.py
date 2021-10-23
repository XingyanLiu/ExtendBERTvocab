# -*- coding: UTF-8 -*-
"""
Extend BERT vocabulary and the corresponding initial embeddings.

@Author: Xingyan Liu
@CreateDate: 2021-09-07
@File: extend_bert_vocab.py
@Project: text_corrector
"""
import os
import re
import sys
from pathlib import Path
from typing import Union, Optional, List, Mapping
import logging
import json
import numpy as np
import tensorflow as tf
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))
import modeling


class Extender(object):
    def __init__(self,
                 bert_dir_in: Union[str, Path],
                 bert_dir_out: Optional[str] = None,
                 n_extend: int = 10,
                 model_name_in: str = 'bert_model',
                 model_name_out: Optional[str] = None,
                 token_fmt: str = "[extended{}]",
                 ):
        """ Extend BERT vocabulary and the corresponding initial embeddings.

        :param bert_dir_in:
            the directory of the original BERT model
        :param bert_dir_out:
            the directory for the extended BERT model
        :param n_extend:
            number of new tokens to be added / extended
        :param model_name_in:
            model prefix
        :param model_name_out:
            file prefix of the extended model
        :param token_fmt:
            the format of the extended tokens
        """
        bert_dir_in = Path(bert_dir_in)
        if bert_dir_out is None:
            bert_dir_out = Path(f"{bert_dir_in}_extended")
        if not os.path.exists(bert_dir_out):
            os.mkdir(bert_dir_out)
        self.bert_dir_in = bert_dir_in
        self.bert_dir_out = bert_dir_out
        self.n_extend = n_extend
        self.model_name_in = model_name_in
        self.model_name_out = model_name_out if model_name_out else model_name_in
        self.token_fmt = token_fmt
        # embedding dimensionality, set when calling `make_bert_config`
        self.n_dim = None
        self.bert_config = None  # self.make_bert_config()

    def run(self):
        """ Main function for extending BERT vocabulary and the corresponding
        initial embeddings.
        """
        self.pad_vocab()
        self.bert_config = self.make_bert_config()

        in_ckpt_prefix = f"{self.bert_dir_in}/{self.model_name_in}.ckpt"
        out_ckpt_prefix = f"{self.bert_dir_out}/{self.model_name_out}.ckpt"

        self.extend_checkpoint(in_ckpt_prefix, out_ckpt_prefix)

    def extend_checkpoint(self, in_ckpt_prefix, out_ckpt_prefix):
        """ Extend BERT initial word embeddings (extend the vocabulary)
        The used initializer refers to: https://github.com/google-research/bert/issues/9

        :param in_ckpt_prefix: str, path-like
            the original BERT model path, may end with '.ckpt'.
            e.g., '../chinese_L-12_H-768_A-12/bert_model.ckpt'
        :param out_ckpt_prefix: str, path-like
            the new extended BERT model path, may end with '.ckpt'.
            e.g., '../chinese_L-12_H-768_A-12_extended/bert_model.ckpt'

        """
        tsr_name_embed = 'bert/embeddings/word_embeddings'
        tsr_name_bias = "cls/predictions/output_bias"
        # step 1: get and extend variables
        tf.reset_default_graph()  # 清空 graph
        with tf.Session() as sess:
            x = tf.get_variable(
                'x_extend', shape=[self.n_extend, self.n_dim],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            sess.run(tf.global_variables_initializer())

            saver0 = tf.train.import_meta_graph(f"{in_ckpt_prefix}.meta")
            saver0.restore(sess, in_ckpt_prefix)

            word_emb = sess.graph.get_tensor_by_name(f'{tsr_name_embed}:0')
            out_bias = sess.graph.get_tensor_by_name(f'{tsr_name_bias}:0')
            np_x = sess.run(x)
            np_word_emb, np_out_bias = sess.run([word_emb, out_bias])
            # concatenation
            np_word_emb_new = np.vstack([np_word_emb, np_x])
            np_out_bias_new = np.hstack([np_out_bias, np.zeros(self.n_extend)])

        # step 2: build a new graph and reset variables
        tf.reset_default_graph()  # 清空上一个图
        with tf.Session() as sess:
            bert_model = self.build_bert_model()  # 这一步可以是其他BERT变体
            sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            # 获取变量名的交集
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(
                    tvars, in_ckpt_prefix)

            # 去除维度对不上的两个变量
            for tsr_name, np_val in zip(
                [tsr_name_embed, tsr_name_bias],
                [np_word_emb_new, np_out_bias_new],
            ):
                if tsr_name in assignment_map.keys():
                    tf.assign(sess.graph.get_tensor_by_name(f'{tsr_name}:0'),
                              np_val)
                    assignment_map.pop(tsr_name)
            # 其余变量从原始的BERT中获取
            tf.train.init_from_checkpoint(in_ckpt_prefix, assignment_map)
            # print(assignment_map.keys())
            saver = tf.train.Saver()
            saver.save(sess, out_ckpt_prefix)

    def make_bert_config(self):
        """ returns the modified BertConfig object """
        config_file_in = f"{self.bert_dir_in}/bert_config.json"
        config_file_out = f"{self.bert_dir_out}/bert_config.json"

        config = modeling.BertConfig.from_json_file(config_file_in)
        self.n_dim = config.hidden_size

        # edit config file
        config.vocab_size = config.vocab_size + self.n_extend
        with open(config_file_out, 'w') as f:
            json.dump(config.to_dict(), f)
        return config

    def pad_vocab(self):
        vocab_new = pad_vocab(
            f"{self.bert_dir_in}/vocab.txt",
            f"{self.bert_dir_out}/vocab.txt",
            n_add=self.n_extend,
            token_fmt=self.token_fmt)

    def build_bert_model(self, max_seq_len=64, **kwargs):
        """ `max_seq_len` does not matter """
        if self.bert_config is None:
            raise ValueError("`.bert_config` has not been properly set, run "
                             "`self.make_bert_config()` first!")
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=tf.placeholder(
                shape=[None, max_seq_len], dtype=tf.int32, name="input_ids"),
            input_mask=tf.placeholder(
                shape=[None, max_seq_len], dtype=tf.int32, name="input_mask"),
            **kwargs
        )
        return bert_model


def pad_vocab(
        vocab_or_file: Union[list, str, Path],
        vocab_file_out: Union[str, Path, None] = None,
        n_add: int = 1,
        token_fmt: str = "[extended{}]",
) -> List[str]:
    """ Padding the vocabulary, with given format `token_fmt`
    :param vocab_or_file:
        input vocabulary list or the filepath (e.g., 'vocab.txt')
    :param vocab_file_out:
    :param n_add:
        the number of extended tokens
    :param token_fmt:
        the extended token format

    :return
    padded vocabulary, a list of strings

    """
    if isinstance(vocab_or_file, (str, Path)):
        with open(vocab_or_file) as f:
            vocab = list(map(lambda s: s.strip(), f.readlines()))
    else:
        vocab = list(vocab_or_file)
    n0 = len(vocab)
    last_tok = vocab[-1]
    if re.fullmatch(token_fmt.replace("{}", "[0-9]+"), last_tok):
        i_start = int(last_tok.strip(token_fmt)) + 1
    else:
        i_start = 0
    vocab = vocab + [token_fmt.format(i) for i in range(i_start, i_start + n_add)]
    assert len(vocab) == (n0 + n_add)

    if vocab_file_out:
        with open(vocab_file_out, 'w') as f:
            f.writelines(map(lambda s: s + '\n', vocab))
    return vocab


def check_tensor_name(ckpt_path_prefix: str, print_info=True):
    """ temporally unused """
    # ckpt_path_prefix = "../chinese_L-12_H-768_A-12/bert_model.ckpt"
    graph = tf.get_default_graph()
    # 获取当前图
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path_prefix))
    saver.restore(sess, ckpt_path_prefix)
    i = 0
    if print_info:
        for tensor in graph.as_graph_def().node:
            print(tensor.name)
            i += 1
            if i >= 100:
                break


def __test__():
    BERT_DIR = "../../PreModels/chinese_L-12_H-768_A-12"
    # in_ckpt_prefix = f'{BERT_DIR}/bert_model.ckpt'
    out_ckpt_prefix = f'{BERT_DIR}_extended/bert_model.ckpt'
    extender = Extender(
        BERT_DIR,
        f"{BERT_DIR}_extest",
        n_extend=10,
        model_name_in="bert_model",
        model_name_out=None,
    )
    extender.run()
    print('----- Checking TF graph -------')
    check_tensor_name(out_ckpt_prefix)


def main():

    # tf.flags.DEFINE_string(
    #       name, default, help, flag_values=.., required=False,)
    tf.flags.DEFINE_integer(
        "random_seed", None, "Random seed for data generation.")
    tf.flags.DEFINE_string(
        "bert_dir_in", None,
        'directory of the original BERT model, e.g., "chinese_L-12_H-768_A-12"',
        required=True,
    )
    tf.flags.DEFINE_string(
        "bert_dir_out", None,
        'directory for the extended (output) BERT model,'
        ' e.g., "../chinese_L-12_H-768_A-12_extended"')
    tf.flags.DEFINE_integer(
        "n_extend", 10, "Number of tokens to be added in the vocabulary",
    )
    tf.flags.DEFINE_string(
        'model_name_in', "bert_model", help='name of the original BERT model')
    tf.flags.DEFINE_string(
        'model_name_out', "bert_model", help='name of the extended BERT model')
    tf.flags.DEFINE_string(
        'token_fmt', "[extended{}]",
        'format of the extended padding tokens default is "[extended{}]"',)
    FLAGS = tf.flags.FLAGS

    extender = Extender(
        FLAGS.bert_dir_in,
        FLAGS.bert_dir_out,
        n_extend=FLAGS.n_extend,
        model_name_in=FLAGS.model_name_in,
        model_name_out=FLAGS.model_name_out,
        token_fmt=FLAGS.token_fmt
    )
    extender.run()


if __name__ == '__main__':
    import time
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    t = time.time()
    # __test__()
    main()
    print('Done running file: {}\nTime: {}'.format(
        os.path.abspath(__file__), time.time() - t,
    ))

