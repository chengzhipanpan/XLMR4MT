# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import six
import socket
import time
import torch
import transformers

from transformers import XLMRobertaForMaskedLM, AutoTokenizer

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.utils as utils
from thumt.utils.context import args_scope
from tqdm import tqdm
from fairseq.utils import new_arange

# from ctcdecode import CTCBeamDecoder

import torch.nn.functional as F

def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output file")
    parser.add_argument("--ptm", type=str, required=True,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Path to prefix parameters")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device_list=[0],
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        decode_batch_size=16,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters.lower())

    return params


def convert_to_string(tensor, tokenizer):
    ids = tensor.tolist()

    s = tokenizer.decode(ids)

    idx = s.find("</s>")

    if idx != -1:
        s = s[:idx]

    idx = s.find("<pad>")

    if idx != -1:
        s = s[:idx]

    s = s.encode("utf-8")

    return s


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def convert_text_to_features(input_sequence, tokenizer, target_text):
    input_sequence = input_sequence.rstrip("<extra_id_0>")
    input_sequence = input_sequence.lstrip()
    tokens = tokenizer.encode_plus(input_sequence)

    input_ids=torch.LongTensor(tokens['input_ids'])[:512].unsqueeze(0)
    eos_pos = input_ids.eq(tokenizer.eos_token_id)
    input_ids[eos_pos] = tokenizer.pad_token_id
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    input_seq = tokenizer.convert_ids_to_tokens(input_ids[0])
    bs_size = input_ids.shape[0]
    
    raw_target_ids = tokenizer.encode_plus(target_text)["input_ids"]
    masked_input_ids = torch.ones(bs_size, 48).long() * tokenizer.mask_token_id
    cur_length = len(raw_target_ids)
    masked_input_ids[:, 0] = tokenizer.bos_token_id

    if cur_length-2 < 48:
        masked_input_ids[:, cur_length-3] = tokenizer.eos_token_id
        masked_input_ids[:, cur_length-2:] = tokenizer.pad_token_id
    else:
        masked_input_ids[:, -1] =  tokenizer.eos_token_id

    features = {
        "source": input_ids.cuda(),
        "source_mask": attention_mask.float().cuda(),
        "masked_input_ids": masked_input_ids.cuda(),
    }

    return features


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def main(args):
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = override_params(params, args)

    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    

    # Create model
    with torch.no_grad():
        # Load configs
        print("Loading XLMR model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.ptm)
        xlmr_model = XLMRobertaForMaskedLM.from_pretrained(args.ptm)
        special_tokens = ["<extra_id_5>", "<extra_id_0>"]
        tokenizer.add_tokens(special_tokens)
        xlmr_model.resize_token_embeddings(len(tokenizer))
        sep_id = tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        model = model_cls(xlmr_model, params)

        params.bos_id = params.sep_id
        params.eos_id = tokenizer.eos_token_id
        params.pad_id = tokenizer.pad_token_id or params.eos_id

        model.load_prefix(args.prefix)

        model = model.cuda()

        if args.half:
            model = model.half()

        xlmr_model.eval()
        model.eval()

        with args_scope(tokenizer=tokenizer):
            input_fn = data.build_input_fn(args.input, "infer", params)
            sorted_key, dataset = input_fn()

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024
        top_beams = params.top_beams
        decode_batch_size = params.decode_batch_size

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([decode_batch_size, top_beams, pad_max]).long()
                  for _ in range(dist.get_world_size())]

        all_outputs = []

        vocab = tokenizer.get_vocab()
        invert_vocab = {}
        for k,v in vocab.items():
            invert_vocab[v] = k
        symbols = []
        for i in range(len(tokenizer)):
            symbols.append(invert_vocab[i])

        from torch.distributions.categorical import Categorical

        def process_one_item(test_text, target_text, f):
            
            special_id = tokenizer.convert_tokens_to_ids(["▁"])[0]
            features = convert_text_to_features(test_text, tokenizer, target_text)
            batch_size = features["source"].shape[0]

            top_pred_string = ""
            original_mask = features["masked_input_ids"]

            past_key_values, denoising_past_key_values = model.get_length(features)
            
            max_steps = 2
            for iter_num in range(max_steps):
                prediction = model.decode_phase1(features, past_key_values)
                
                prediction = prediction.squeeze(0)
                top_pred = torch.argmax(prediction, dim=-1)

                pred_prob = F.softmax(prediction, dim=-1)
                pred_prob = torch.max(pred_prob, dim=-1).values

                top_pred_toks = tokenizer.convert_ids_to_tokens(top_pred)
                top_pred_string = tokenizer.convert_tokens_to_string(top_pred_toks)

                masked_input = tokenizer.convert_ids_to_tokens(features["masked_input_ids"].squeeze(0))
                top_pred = top_pred.unsqueeze(0)
                top_score = torch.softmax(prediction, dim=-1)
                top_score = torch.max(top_score, dim=-1)[0].unsqueeze(0)

                if (iter_num + 1) <= max_steps:
                    pred_score = F.softmax(prediction, dim=-1)
                    
                    t_prediction = prediction.unsqueeze(0)
                    # print(t_pred_score.shape)
                    pred_score = torch.max(pred_score, dim=-1)[0]
                    pred_score = pred_score.unsqueeze(0)

                    skeptical_mask = _skeptical_unmasking(
                        pred_score, top_pred.ne(tokenizer.pad_token_id), 1 - (iter_num + 1) / max_steps
                    )

                    pad_mask = top_pred.eq(tokenizer.pad_token_id)
                    eos_mask = top_pred.eq(tokenizer.eos_token_id)
                    bad_tok_mask = top_pred.eq(len(tokenizer)-1) | top_pred.eq(len(tokenizer)-2)
                    pad_mask = pad_mask | eos_mask | bad_tok_mask

                    prev_mask = features["masked_input_ids"].eq(tokenizer.mask_token_id)

                    features["masked_input_ids"] = top_pred.masked_fill(~prev_mask, 0) + features["masked_input_ids"].masked_fill(prev_mask, 0)
                    top_pred_toks = tokenizer.convert_ids_to_tokens(features["masked_input_ids"].squeeze(0))
                    top_pred_string = tokenizer.convert_tokens_to_string(top_pred_toks)

                    masked_indices = skeptical_mask
                    masked_indices = masked_indices & prev_mask
                    
                    features["masked_input_ids"].masked_fill_(masked_indices, tokenizer.mask_token_id)
                    
                raw_predictions = features["masked_input_ids"].clone()

                raw_predictions = raw_predictions.unsqueeze(-1)
                raw_predictions = raw_predictions.repeat(1, 1, 2)
                raw_predictions = raw_predictions.reshape(1, -1)

                features["denoising_masked_input_ids"] = raw_predictions

                prediction = model.denoise_phase(features, denoising_past_key_values)                
                output_probs = F.softmax(prediction, dim=-1)
                prediction = prediction.squeeze(0)
                output_probs = output_probs.squeeze(0)
                prob = torch.max(output_probs, dim=-1).values
                top_pred = torch.argmax(prediction, dim=-1)

                second_mask = (prob < 0.6) & (top_pred.eq(special_id))
                vals, indices = prediction.topk(k=2, dim=1, largest=True, sorted=True)
                top_pred[second_mask] = indices[:, 1][second_mask] 

                top_pred_toks = tokenizer.convert_ids_to_tokens(top_pred)
                _toks = top_pred.int().tolist()
                _toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
                ctc_result = top_pred.new_tensor([v for v in _toks if v != special_id])
                top_pred_toks = tokenizer.convert_ids_to_tokens(ctc_result)

                top_pred_string = tokenizer.convert_tokens_to_string(top_pred_toks)

                masked_input = tokenizer.convert_ids_to_tokens(features["masked_input_ids"].squeeze(0))

            return top_pred_string

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        with open(args.input, "r") as fin, open(os.path.join(args.output, "./references.txt"), "w") as f_refer, open(os.path.join(args.output, "./hypo.txt"), "w") as f_hypo, open("./test_log.log", "w") as flog:
            for test_text in tqdm(fin):
                test_text = test_text.rstrip("\n")
                src, tgt = str.split(test_text, "<extra_id_0>")
                print(tgt, file=f_refer)
                cur_tgt = "<s> " + tgt + " </s>"
                result = process_one_item(src, cur_tgt, flog)
                print(result, file=f_hypo)
            

# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
