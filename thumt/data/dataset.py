# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tensorflow as tf
import operator

from thumt.utils.context import get_args

from transformers import AutoTokenizer
import random

xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
special_tokens = ["<extra_id_5>", "<extra_id_0>"]
xlmr_tok.add_tokens(special_tokens)
spec_tok_ids = xlmr_tok.convert_tokens_to_ids(special_tokens)



def sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, sorted_inputs


def _pad(x):
    max_len = 0
    masks = []

    pad = xlmr_tok.pad_token_id

    for ids in x:
        max_len = max(len(ids), max_len)

    for ids in x:
        mask = []

        for _ in range(len(ids)):
            mask.append(1)

        for _ in range(max_len - len(ids)):
            ids.append(pad)
            mask.append(0)

        masks.append(mask)

    return x, masks


def _pad_target(x, given_len):
    max_len = given_len
    masks = []
    result_x = []

    pad = xlmr_tok.pad_token_id

    for ids in x:
        mask = []

        if max_len <= len(ids):
            mask = [1] * max_len
            ids = ids[:max_len]
        else:
            for _ in range(len(ids)):
                mask.append(1)

            for _ in range(max_len - len(ids)):
                ids.append(pad)
                mask.append(0)

        masks.append(mask)
        result_x.append(ids)

    return result_x, masks


def length_to_mask(lens, max_len):
    return torch.arange(max_len).expand(len(lens), max_len).cpu() < lens.unsqueeze(1)


def build_input_fn(filenames, mode, params):
    def train_input_fn(path, tokenizer, params):
        dataset = tf.data.TextLineDataset(path)
        dataset = dataset.repeat(None)
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        def py_tokenize(x):
            x = tokenizer.encode(x.numpy().decode("utf-8", errors="ignore"))
            return tf.convert_to_tensor(x, dtype=tf.int32)

        def map_func(x):
            return tf.py_function(py_tokenize, [x], tf.int32)

        dataset = dataset.map(map_func)

        dataset = dataset.map(
            lambda x: {
                "inputs": x,
                "lengths": tf.shape(x)[0],
            },
            num_parallel_calls=tf.data.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x):
            return x["lengths"]

        def valid_size(x):
            size = element_length_func(x)
            return tf.logical_and(size >= min_length, size <= max_length)

        transformation_fn = tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            boundaries,
            batch_sizes,
            padded_shapes={
                "inputs": tf.TensorShape([None]),
                "lengths": tf.TensorShape([]),
                },
            padding_values={
                "inputs": 0,
                "lengths": 0,
                },
            pad_to_bucket_boundary=False)

        dataset = dataset.filter(valid_size)
        dataset = dataset.apply(transformation_fn)

        return dataset

    def infer_input_fn():
        sorted_key, sorted_data = sort_input_file(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(sorted_data))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        args = get_args()
        tokenizer = args.tokenizer

        def py_tokenize(x):
            x = tokenizer.encode(x.numpy().decode("utf-8", errors="ignore"))[:-1]
            return tf.convert_to_tensor(x, dtype=tf.int32)

        def map_func(x):
            return tf.py_function(py_tokenize, [x], tf.int32)

        dataset = dataset.map(map_func)

        dataset = dataset.map(
            lambda x: {
                "source": x,
                "source_length": tf.shape(x)[0],
            },
            num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes={
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([])
            },
            padding_values={
                "source": 0,
                "source_length": 0
            })

        return sorted_key, dataset

    if mode == "train":
        return train_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode %s" % mode)


def _random_mask(ori_inputs, prob):
    pad = xlmr_tok.pad_token_id
    bos = xlmr_tok.bos_token_id
    eos = xlmr_tok.eos_token_id

    inputs = ori_inputs.clone()
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, prob)

    special_tokens_mask = inputs.eq(pad) | inputs.eq(bos) | inputs.eq(eos)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 60% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = xlmr_tok.convert_tokens_to_ids(xlmr_tok.mask_token)

    return inputs, labels


def to_translation_features(features, sep_id, mode="train"):

    pad = xlmr_tok.pad_token_id

    if mode == "train":
        inputs = features["inputs"]
        lengths = features["lengths"]
        sources = []
        targets = []

        inputs = inputs.numpy().tolist()
        lengths = lengths.numpy().tolist()

        for toks, length in zip(inputs, lengths):
            flag = 0
            source = []
            target = []

            for i in range(length):
                if toks[i] == sep_id:
                    flag = 1

                if flag == 0:
                    source.append(toks[i])
                else:
                    target.append(toks[i])

            sources.append(source)
            targets.append(target)

        sources, source_masks = _pad(sources)
        source = torch.LongTensor(sources).cuda()
        source_mask = torch.FloatTensor(source_masks).cuda()

        # Targets we may need to copy one, and mask all of them ~ Good, that's our solution

        given_len = 2 * len(sources[0])
        targets, target_masks = _pad_target(targets, given_len)
        target_mask = torch.FloatTensor(target_masks).cuda()
        raw_target = torch.LongTensor(targets).cuda()
        raw_target[:, 0] = 0

        target_lengths = raw_target.ne(pad).sum(dim=-1).cuda()
        
        prob = max(0.2, random.random())
        denoising_masked_input = torch.ones_like(raw_target) * xlmr_tok.eos_token_id
        # denoising_masked_input = wrapper_ctc(raw_target, xlmr_tok, denoising_masked_input)
        
        denoising_masked_input[:, 0] = xlmr_tok.bos_token_id
        denoising_target_mask = denoising_masked_input
        denoising_target = denoising_masked_input
 
        # denoising_target[:, 0] = 0
        denoising_masked_input.cuda()
        denoising_target.cuda()
        denoising_target_mask.cuda()
        # denoising_target_mask[:, 0] = 1
        
        ###########################################################
        ################### Full Masked Input #####################
        ###########################################################

        pad_mask = raw_target.eq(pad)
        full_masked_input = torch.ones_like(raw_target) * xlmr_tok.mask_token_id
        full_masked_target = raw_target

        full_masked_target_mask = (full_masked_target.ne(-100) & full_masked_target.ne(0) & full_masked_target.ne(pad)).float().cuda()
        full_masked_target_mask[:, 0] = 1

        prob = random.random()
        # full_masked_input, _ = _random_mask(raw_target, prob)

        full_masked_input.cuda()
        full_masked_target.cuda()
        full_masked_target_mask.cuda()

        features = {
            "source": source,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "full_masked_input_ids": full_masked_input,
            "full_masked_target": full_masked_target,
            "full_masked_target_mask": full_masked_target_mask,
            "denoising_masked_input_ids": denoising_masked_input,
            "denoising_target": denoising_target,
            "denoising_target_mask": denoising_target_mask,
            "target_lengths": target_lengths,
        }

        return features, raw_target, denoising_target



def to_translation_features_v5(features, sep_id, mode="train"):

    pad = xlmr_tok.pad_token_id

    if mode == "train":
        inputs = features["inputs"]
        lengths = features["lengths"]
        sources = []
        targets = []

        inputs = inputs.numpy().tolist()
        lengths = lengths.numpy().tolist()

        for toks, length in zip(inputs, lengths):
            flag = 0
            source = []
            target = []

            for i in range(length):
                if toks[i] == sep_id:
                    flag = 1

                if flag == 0:
                    source.append(toks[i])
                else:
                    target.append(toks[i])

            sources.append(source)
            targets.append(target)

        sources, source_masks = _pad(sources)
        source = torch.LongTensor(sources).cuda()
        source_mask = torch.FloatTensor(source_masks).cuda()

        given_len = 48
        targets, target_masks = _pad_target(targets, given_len)
        target_mask = torch.FloatTensor(target_masks).cuda()
        raw_target = torch.LongTensor(targets).cuda()

        raw_target[:, 0] = 0

        target_lengths = raw_target.ne(pad).sum(dim=-1).cuda()
        
        denoising_masked_input = torch.ones_like(raw_target) * xlmr_tok.eos_token_id
        # denoising_masked_input = wrapper_ctc(raw_target, xlmr_tok, denoising_masked_input)
        
        denoising_masked_input[:, 0] = xlmr_tok.bos_token_id
        denoising_target_mask = denoising_masked_input
        denoising_target = denoising_masked_input
 
        # denoising_target[:, 0] = 0
        denoising_masked_input.cuda()
        denoising_target.cuda()
        denoising_target_mask.cuda()
        # denoising_target_mask[:, 0] = 1
        
        ###########################################################
        ################### Full Masked Input #####################
        ###########################################################

        pad_mask = raw_target.eq(pad)
        full_masked_input = torch.ones_like(raw_target) * xlmr_tok.mask_token_id
        full_masked_target = raw_target

        full_masked_target_mask = (full_masked_target.ne(-100) & full_masked_target.ne(0) & full_masked_target.ne(pad)).float().cuda()
        full_masked_target_mask[:, 0] = 1

        prob = max(0.2, random.random())
        full_masked_input, _ = _random_mask(raw_target, prob)

        full_masked_input.cuda()
        full_masked_target.cuda()
        full_masked_target_mask.cuda()

        features = {
            "source": source,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "full_masked_input_ids": full_masked_input,
            "full_masked_target": full_masked_target,
            "full_masked_target_mask": full_masked_target_mask,
            "denoising_masked_input_ids": denoising_masked_input,
            "denoising_target": denoising_target,
            "denoising_target_mask": denoising_target_mask,
            "target_lengths": target_lengths,
        }

        return features, raw_target, denoising_target



def to_QG_features(features, sep_id, mode="train"):

    pad = xlmr_tok.pad_token_id

    if mode == "train":
        inputs = features["inputs"]
        lengths = features["lengths"]
        sources = []
        targets = []

        inputs = inputs.numpy().tolist()
        lengths = lengths.numpy().tolist()

        for toks, length in zip(inputs, lengths):
            flag = 0
            source = []
            target = []

            for i in range(length):
                if toks[i] == sep_id:
                    flag = 1

                if flag == 0:
                    source.append(toks[i])
                else:
                    target.append(toks[i])

            sources.append(source)
            targets.append(target)

        sources, source_masks = _pad(sources)
        source = torch.LongTensor(sources).cuda()
        source_mask = torch.FloatTensor(source_masks).cuda()

        # Targets we may need to copy one, and mask all of them ~ Good, that's our solution

        given_len = 48 # 2 * len(sources[0])
        targets, target_masks = _pad_target(targets, given_len)
        target_mask = torch.FloatTensor(target_masks).cuda()
        raw_target = torch.LongTensor(targets).cuda()

        raw_target[:, 0] = 0

        target_lengths = raw_target.ne(pad).sum(dim=-1).cuda()
        
        prob = random.random()
        # 也用padding把
        denoising_masked_input = torch.ones_like(raw_target) * xlmr_tok.eos_token_id
        # denoising_masked_input = wrapper_ctc(raw_target, xlmr_tok, denoising_masked_input)
        
        denoising_masked_input[:, 0] = xlmr_tok.bos_token_id
        denoising_target_mask = denoising_masked_input
        denoising_target = denoising_masked_input
 
        # denoising_target[:, 0] = 0
        denoising_masked_input.cuda()
        denoising_target.cuda()
        denoising_target_mask.cuda()
        # denoising_target_mask[:, 0] = 1
        
        ###########################################################
        ################### Full Masked Input #####################
        ###########################################################

        pad_mask = raw_target.eq(pad)
        full_masked_input = torch.ones_like(raw_target) * xlmr_tok.mask_token_id
        full_masked_target = raw_target

        prob = max(0.2, random.random())
        full_masked_input, _ = _random_mask(raw_target, prob)

        full_masked_target_mask = (full_masked_target.ne(-100) & full_masked_target.ne(0) & full_masked_target.ne(pad)).float().cuda()
        full_masked_target_mask[:, 0] = 1

        full_masked_input.cuda()
        full_masked_target.cuda()
        full_masked_target_mask.cuda()

        features = {
            "source": source,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "full_masked_input_ids": full_masked_input,
            "full_masked_target": full_masked_target,
            "full_masked_target_mask": full_masked_target_mask,
            "denoising_masked_input_ids": denoising_masked_input,
            "denoising_target": denoising_target,
            "denoising_target_mask": denoising_target_mask,
            "target_lengths": target_lengths,
        }

        return features, raw_target, denoising_target
