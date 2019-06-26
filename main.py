# -*- coding: utf-8 -*-

"""
Copyright 2019 Tae Hwan Jung

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data_utils
import argparse

import torch
from pytorch_pretrained_bert import BertTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch XLNet Language Model')
    parser.add_argument('--data', type=str, default='data.txt')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Path to the sentence piece model from pytorch-pretrained-BERT')
    parser.add_argument('--bsz', type=int, default=1, help="batch size")
    parser.add_argument('--seq_len', type=int, default=512, help="Sequence length.")
    parser.add_argument('--reuse_len', type=int, default=256,
                        help="Number of token that can be reused as memory. "
                             "Could be half of `seq_len`.")
    parser.add_argument('--perm_size', type=int,
                        default=256,
                        help="the length of longest permutation. Could be set to be reuse_len.")
    parser.add_argument('--bi_data', type=bool, default=True,
                        help="whether to create bidirectional data")
    parser.add_argument('--mask_alpha', type=int,
                        default=6, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--num_predict', type=int,
                        default=85, help="Num of tokens to predict.")

    args = parser.parse_args()

    sp = BertTokenizer.from_pretrained(args.tokenizer)

    feature = data_utils._create_data(sp=sp,
                            input_paths=args.data,
                            seq_len=args.seq_len,
                            reuse_len=args.reuse_len,
                            bi_data=args.bi_data,
                            num_predict=args.num_predict,
                            mask_alpha=args.mask_alpha,
                            mask_beta=args.mask_beta)

    permutation = data_utils.make_permute(feature,
                                          reuse_len=args.reuse_len,
                                          seq_len=args.seq_len,
                                          perm_size=args.perm_size,
                                          num_predict=args.num_predict)

