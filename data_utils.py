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

import random

import torch
import numpy as np

special_symbols = {
    "[UNK]"  : 0,
    "[CLS]"  : 1,
    "[SEP]"  : 2,
    "[PAD]"  : 3,
    "[MASK]" : 4,
}
UNK_ID = special_symbols["[UNK]"]
CLS_ID = special_symbols["[CLS]"]
SEP_ID = special_symbols["[SEP]"]
MASK_ID = special_symbols["[MASK]"]

def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    if begin_idx + tot_len >= data_len:
        print("[_split_a_and_b] returns None: "
                "begin_idx %d + tot_len %d >= data_len %d",
                begin_idx, tot_len, data_len)
        return None

    end_idx = begin_idx + 1
    cut_points = []
    while end_idx < data_len:
        if sent_ids[end_idx] != sent_ids[end_idx - 1]:
            if end_idx - begin_idx >= tot_len: break
            cut_points.append(end_idx)
        end_idx += 1

    a_begin = begin_idx
    if len(cut_points) == 0 or random.random() < 0.5:
        # NotNext
        label = 0
        if len(cut_points) == 0:
            a_end = end_idx
        else:
            a_end = random.choice(cut_points)

        b_len = max(1, tot_len - (a_end - a_begin))
        # (zihang): `data_len - 1` to account for extend_target
        b_begin = random.randint(0, data_len - 1 - b_len)
        b_end = b_begin + b_len
        while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
            b_begin -= 1
        # (zihang): `data_len - 1` to account for extend_target
        while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
            b_end += 1

        new_begin = a_end
    else:
        # isNext
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx

        new_begin = b_end

    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            # delete the right side only for the LM objective
            a_end -= 1
        else:
            b_end -= 1

    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        if a_end >= data_len or b_end >= data_len:
            print("[_split_a_and_b] returns None: "
                          "a_end %d or b_end %d >= data_len %d",
                          a_end, b_end, data_len)
            return None
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])

    return ret

def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False

def _sample_mask(sp, seg, mask_alpha, mask_beta,
                 reverse=False, max_gram=5, goal_num_predict=None):
    """Sample `goal_num_predict` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    if reverse:
        seg = np.flip(seg, 0)

    cur_len = 0
    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx
        while beg < seg_len and not _is_start_piece(sp.convert_ids_to_tokens([seg[beg].item()])):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece(sp.convert_ids_to_tokens([seg[beg].item()])):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    if reverse:
        mask = np.flip(mask, 0)

    return mask

def _create_data(sp, input_paths, seq_len, reuse_len,
                bi_data, num_predict, mask_alpha, mask_beta):
    features = []

    f = open(input_paths, 'r')
    lines = f.readlines()
    input_data, sent_ids, sent_id = [], [], True

    for line in lines:
        tokens = sp.tokenize(line)
        cur_sent = sp.convert_tokens_to_ids(tokens)
        input_data.extend(cur_sent)
        sent_ids.extend([sent_id] * len(cur_sent))
        sent_id = not sent_id

    # shape of data : [1, 582]
    data = np.array([input_data], dtype=np.int64)
    sent_ids = np.array([sent_ids], dtype=np.bool)

    assert reuse_len < seq_len - 3

    data_len = data.shape[1]
    sep_array = np.array([SEP_ID], dtype=np.int64)
    cls_array = np.array([CLS_ID], dtype=np.int64)

    i = 0
    while i + seq_len <= data_len:
        inp = data[0, i: i + reuse_len]
        tgt = data[0, i + 1: i + reuse_len + 1]

        results = _split_a_and_b(
            data[0], # all line in one Text file.
            sent_ids[0],
            begin_idx=i + reuse_len,
            tot_len=seq_len - reuse_len - 3,
            extend_target=True)

        # unpack the results
        (a_data, b_data, label, _, a_target, b_target) = tuple(results)

        # sample ngram spans to predict
        reverse = bi_data
        if num_predict is None:
            num_predict_0 = num_predict_1 = None
        else:
            num_predict_1 = num_predict // 2
            num_predict_0 = num_predict - num_predict_1

        mask_0 = _sample_mask(sp, inp, mask_alpha, mask_beta, reverse=reverse,
                              goal_num_predict=num_predict_0)
        mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                  sep_array, cls_array]),
                              mask_alpha, mask_beta,
                              reverse=reverse, goal_num_predict=num_predict_1)

        # concatenate data
        cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                   sep_array, cls_array])
        seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                  [1] * b_data.shape[0] + [1] + [2])
        assert cat_data.shape[0] == seq_len
        assert mask_0.shape[0] == seq_len // 2
        assert mask_1.shape[0] == seq_len // 2

        # the last two CLS's are not used, just for padding purposes
        tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
        assert tgt.shape[0] == seq_len

        is_masked = np.concatenate([mask_0, mask_1], 0)
        if num_predict is not None:
            assert np.sum(is_masked) == num_predict

        feature = {
            "input": cat_data,
            "is_masked": is_masked,
            "target": tgt,
            "seg_id": seg_id,
            "label": [label],
        }
        features.append(feature)

        i += reuse_len

    f.close()
    return features

def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.

    Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    targets: int64 Tensor in shape [seq_len], target ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected
      for partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.
    """

    # Generate permutation indices
    index = torch.arange(seq_len, dtype=torch.int64)

    index = torch.reshape(index, [-1, perm_size]).t()
    index = index[torch.randperm(index.shape[0])]
    index = torch.reshape(index.t(), [-1])

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = ~(torch.eq(inputs, SEP_ID) | torch.eq(inputs, CLS_ID))
    non_mask_tokens = (~is_masked) & non_func_tokens
    masked_or_func_tokens = ~non_mask_tokens

    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -torch.ones([seq_len], dtype=torch.int64)

    # put -1 if `non_mask_tokens(real token not cls or sep)` not permutation index
    rev_index = torch.where(non_mask_tokens, smallest_index, index)

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = masked_or_func_tokens & non_func_tokens
    target_mask = target_tokens.type(torch.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
    self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) &  masked_or_func_tokens
    perm_mask = perm_mask.type(torch.float32)

    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = torch.cat([inputs[0: 1], targets[: -1]], dim=0)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q

def make_permute(feature, reuse_len, seq_len, perm_size, num_predict):

    inputs = torch.LongTensor(feature.pop("input"))
    target = torch.LongTensor(feature.pop("target"))
    is_masked = torch.ByteTensor(feature.pop("is_masked"))

    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len], # inp
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:], # (senA, seq, senBm seq, cls)
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)

    perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len])],
                            dim=1)
    perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len]), perm_mask_1],
                            dim=1)

    perm_mask = torch.cat([perm_mask_0, perm_mask_1], dim=0)
    target = torch.cat([target_0, target_1], dim=0)
    target_mask = torch.cat([target_mask_0, target_mask_1], dim=0)
    input_k = torch.cat([input_k_0, input_k_1], dim=0)
    input_q = torch.cat([input_q_0, input_q_1], dim=0)

    if num_predict is not None:
        indices = torch.arange(seq_len, dtype=torch.int64)
        bool_target_mask = target_mask.byte()
        indices = indices[bool_target_mask]

        ##### extra padding due to CLS/SEP introduced after prepro
        actual_num_predict = indices.shape[0]
        pad_len = num_predict - actual_num_predict

        assert seq_len >= actual_num_predict

        ##### target_mapping
        target_mapping = torch.eye(seq_len, dtype=torch.float32)[indices]
        paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
        target_mapping = torch.cat([target_mapping, paddings], dim=0)
        feature["target_mapping"] = torch.reshape(target_mapping,
                                                [num_predict, seq_len])
        ##### target
        target = target[bool_target_mask]
        paddings = torch.zeros([pad_len], dtype=target.dtype)
        target = torch.cat([target, paddings], dim=0)
        feature["target"] = torch.reshape(target, [num_predict])

        ##### target mask
        target_mask = torch.cat(
            [torch.ones([actual_num_predict], dtype=torch.float32),
             torch.zeros([pad_len], dtype=torch.float32)],
            dim=0)
        feature["target_mask"] = torch.reshape(target_mask, [num_predict])
    else:
        feature["target"] = torch.reshape(target, [seq_len])
        feature["target_mask"] = torch.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    feature["seg_id"] = torch.IntTensor(feature["seg_id"])
    feature["perm_mask"] = torch.reshape(perm_mask, [seq_len, seq_len])
    feature["input_k"] = torch.reshape(input_k, [seq_len])
    feature["input_q"] = torch.reshape(input_q, [seq_len])

    return feature