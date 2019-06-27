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

import xlnet
import torch
from pytorch_pretrained_bert import BertTokenizer

def two_stream_loss(features, labels, mems, is_training):
    """Pretraining loss with two-stream attention Transformer-XL."""

    #### Unpack input
    mem_name = "mems"
    mems = mems.get(mem_name, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch XLNet Language Model')
    parser.add_argument('--data', type=str, default='data.txt')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Path to the sentence piece model from pytorch-pretrained-BERT')
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
    parser.add_argument('--mem_len', type=int,
                        default=384, help="Number of steps to cache")

    args = parser.parse_args()

    sp = BertTokenizer.from_pretrained(args.tokenizer)
    model = xlnet.XLNet(n_token=len(sp.vocab), n_layer=6, n_head=4, d_head=8,
                        d_inner=32, d_model=32,
                        dropout=0.0, dropatt=0.0,
                        attn_type="bi", bi_data=args.bi_data,
                        clamp_len=-1, same_length=False,
                        reuse_len=args.reuse_len, mem_len=args.mem_len)

    features = data_utils._create_data(sp=sp,
                            input_paths=args.data,
                            seq_len=args.seq_len,
                            reuse_len=args.reuse_len,
                            bi_data=args.bi_data,
                            num_predict=args.num_predict,
                            mask_alpha=args.mask_alpha,
                            mask_beta=args.mask_beta)

    for feature in features:
        permutation = data_utils.make_permute(feature,
                                              reuse_len=args.reuse_len,
                                              seq_len=args.seq_len,
                                              perm_size=args.perm_size,
                                              num_predict=args.num_predict)

        # batch size is 1
        inp_k = permutation['input_k'].unsqueeze(-1) # [seq_len, 1(=bsz)]
        seg_id = permutation['seg_id'].unsqueeze(-1) # [seq_len, 1(=bsz)]
        input_mask = None
        mems = None
        perm_mask = permutation['perm_mask'].unsqueeze(-1) # [seq_len, seq_len, 1(=bsz)]
        target_mapping = \
            permutation['target_mapping'].unsqueeze(-1) # [num_predict, seq_len, 1(=bsz)]
        inp_q = permutation['input_q'].unsqueeze(-1) # [seq_len, 1(=bsz)]

        output, new_mems, lookup_table = model(inp_k=inp_k, seg_id=seg_id, input_mask=input_mask,
              mems=mems, perm_mask=perm_mask,
              target_mapping=target_mapping, inp_q=inp_q)