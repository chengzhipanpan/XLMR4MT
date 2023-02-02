# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F

import thumt.utils as utils
import thumt.modules as modules
import torch.distributed as dist
import thumt.utils.summary as summary
import thumt.data.dataset as dataset
from torch.distributions.categorical import Categorical

from transformers import AutoTokenizer
import random

from fairseq.utils import new_arange



def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(*new_shape)
    return tensor.permute(0, 2, 1, 3)


def _combine_heads(x):
    batch = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    channels = x.shape[3]

    y = torch.transpose(x, 2, 1)

    return torch.reshape(y, [batch, length, heads * channels])


def _random_mask(ori_inputs, tokenizer, prob):
    pad = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    special_id = tokenizer.convert_tokens_to_ids(["▁"])[0]
    mask_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]

    inputs = ori_inputs.clone()
    probability_matrix = torch.full(inputs.shape, prob)
    
    special_tokens_mask = (inputs.eq(bos) | inputs.eq(pad) | inputs.eq(eos))

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # # 60% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.4)).bool() & masked_indices
    inputs[indices_replaced] = special_id
    
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # 20% of the time, we replace masked input tokens with one repetition from current sentence
    indices_repetition = masked_indices & ~indices_random & ~indices_replaced
    repetition_idx = torch.randint(indices_repetition.size(-1), inputs[0].shape, dtype=torch.long)
    
    repetition_words = ori_inputs.index_select(1, repetition_idx)
    inputs[indices_repetition] = repetition_words[indices_repetition]

    return inputs


class Prompt(modules.Module):

    def __init__(self, model, num_prompts, prompt_length, name="prompt"):
        super(Prompt, self).__init__(name=name)

        self.embed_dim = model.config.hidden_size
        self.split_size = self.embed_dim
        self.hidden_size = model.config.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scales = nn.Parameter(
            torch.ones([num_prompts]))
        self.add_name(self.scales, "scales")
        self.prompts = nn.Parameter(
            torch.empty(
            [
                num_prompts, 2 * self.num_decoder_layers,
                prompt_length, self.hidden_size
            ]))
        self.add_name(self.prompts, "prompts")
        self._model = [model]

        with torch.no_grad():
            for i in range(self.prompts.shape[0]):
                for j in range(self.prompts.shape[1]):
                    nn.init.xavier_uniform_(self.prompts[i, j])

    @property
    def model(self):
        return self._model[0]

    def forward(self, batch_size):
        key_values = [[] for i in range(self.prompts.shape[0])]

        k_list = []
        v_list = []
        for i in range(self.prompts.shape[0]):
            for j in range(self.num_decoder_layers):
                scale = torch.maximum(torch.ones([]), self.scales[i])
                k = self.prompts[i, 2*j][None, :, :] * scale
                v = self.prompts[i, 2*j+1][None, :, :] * scale
                k = k.repeat([batch_size, 1, 1])
                v = v.repeat([batch_size, 1, 1])
                k = _split_heads(k, self.num_heads, self.head_dim)
                v = _split_heads(v, self.num_heads, self.head_dim)
                key_values[i].append(torch.cat([k.unsqueeze(0), v.unsqueeze(0)], dim=0))
       
        return key_values


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


class mXLMR_SGA(modules.Module):
    def __init__(self, model, params, name="mxlmr_SGA"):
        super(mXLMR_SGA, self).__init__(name=name)
        self.params = params
        self._xlmr_model = [model.roberta]
        # self.xlmr_model = model

        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        special_tokens = ["<extra_id_5>", "<extra_id_0>"]
        self.xlmr_tok.add_tokens(special_tokens)

        # Mask id. the first masked is the outputs that we need.
        self.sep_id = 250001 # len(self.xlmr_tok) - 1
        self.blank_idx = self.xlmr_tok.convert_tokens_to_ids(["▁"])[0]



        params.hidden_size = model.config.hidden_size

        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.k_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self.v_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)

        self._classifier = [model.lm_head]

        if params.share_prompt:
            self.prompt_model = Prompt(model, 1, params.prompt_length)
        else:
            self.prompt_model = Prompt(model, 2+params.re_encoding,
                                       params.prompt_length)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

    @property
    def xlmr_model(self):
        return self._xlmr_model[0]

    @property
    def classifier(self):
        return self._classifier[0]

    @property
    def src_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def tgt_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def softmax_embedding(self):
        return self.tgt_embedding

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])
    
    def get_normalized_probs(self, logits, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # logits = logits.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def decode_phase1(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        # target_pos = (input_ids[0] == self.sep_id).nonzero(as_tuple=True)[0][0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )

        #############################################################
        ########### Semantic Guided Translation Prompt   ############
        #############################################################

        semantic_emb = outputs.hidden_states[1:]

        # print(outputs.hidden_states[-1].requires_grad)

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[1])]

        new_prefix_attention_mask = torch.ones(batch_size, semantic_emb[0].shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)

        return logits, past_key_values, new_sem_emb_concat

    
    def denoise_phrase(self, features, past_key_values, new_sem_emb_concat):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        attention_mask = src_mask
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        new_prefix_attention_mask = torch.ones(batch_size, features["source"].shape[1]).to(input_ids.device)
        masked_attention_mask = torch.ones(batch_size, features["denoising_masked_input_ids"].shape[1]).to(input_ids.device)
        
        new_attention_mask = torch.cat((attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        # #############################################################
        # ############# Semantic Guided Denoising Prompt   ############
        # #############################################################

        final_masked_input_ids =  features["denoising_masked_input_ids"]
        input_shape = final_masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        denoising_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[2])]
        denoising_outputs = self.xlmr_model(
            input_ids=final_masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=denoising_past_key_values,
            output_hidden_states=True,
        )

        denoised_logits = self.classifier(denoising_outputs.last_hidden_state)

        return denoised_logits

    
    def encode_semantic(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )

        semantic_emb = outputs.hidden_states[1:]

        # print(outputs.hidden_states[-1].requires_grad)

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]

        return past_key_values, new_sem_emb_concat
    
    def get_translation_outputs(self, translation_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, translation_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["full_masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)
        return logits

    def get_denoise_outputs(self, denoising_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)

        input_shape = input_ids.shape
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, denoising_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["denoising_masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        
        denoising_logits = self.classifier(translation_outputs.last_hidden_state)
        return denoising_logits

    @torch.inference_mode()
    def get_unroll_prediction(self, denoising_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)

        input_shape = input_ids.shape
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, denoising_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["unroll_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        
        denoising_logits = self.classifier(translation_outputs.last_hidden_state)
        return denoising_logits.detach()
        

    def forward(self, features, labels, denoised_labels, mode="sundae"):
        past_key_values, new_sem_emb_concat = self.encode_semantic(features)
        stage1_logits = self.get_translation_outputs(past_key_values[1], new_sem_emb_concat, features)
        stage1_logits = stage1_logits[:, :labels.shape[1], :]


        all_stage2_logits = []

        # A random variable used in two stages 
        start_random = random.random() * 0.2

        clone_stage1_logits = stage1_logits.clone().detach()

        stage1_prob = self.get_normalized_probs(clone_stage1_logits, log_probs=False)
        prob = torch.max(stage1_prob, dim=-1).values
        vals, indices = stage1_prob.topk(k=2, dim=-1, largest=True, sorted=True)

        stage2_inputs = torch.argmax(clone_stage1_logits, dim=-1)
        second_mask = (stage2_inputs.eq(self.blank_idx))
        second_mask = second_mask.to(stage2_inputs.device)

        # unroll inputs up sampling -- information need to be provided for these tokens, as they are not trustable
        unroll_inputs = stage2_inputs # torch.argmax(clone_stage1_logits, dim=-1) # Categorical(logits=clone_stage1_logits).sample()
        indices = indices.to(unroll_inputs.device)
        unroll_inputs[second_mask] = indices[:, :, 1][second_mask]

        # random noise injection, to provide some noise -- may just need very small noises in it
        unroll_inputs = _random_mask(unroll_inputs.clone(), self.xlmr_tok, start_random)
        start_random = start_random / 2

        features["unroll_input_ids"] = unroll_inputs

        if random.random() < 0.01:
            verbose = True
        else:
            verbose = False
        
        max_steps = 1
        for _ in range(max_steps):
            features["denoising_masked_input_ids"] = unroll_inputs.clone()

            stage2_logits = self.get_denoise_outputs(past_key_values[2], new_sem_emb_concat, features)
            stage2_logits = stage2_logits[:, :denoised_labels.shape[1], :]

            if _ < max_steps - 1:
                sampled_unroll_logits = self.get_unroll_prediction(past_key_values[2], new_sem_emb_concat, features)

                stage1_prob = self.get_normalized_probs(sampled_unroll_logits, log_probs=False)
                prob = torch.max(stage1_prob, dim=-1).values
                vals, indices = stage1_prob.topk(k=2, dim=-1, largest=True, sorted=True)

                stage2_inputs = torch.argmax(stage1_prob, dim=-1)
                second_mask = (stage2_inputs.eq(self.blank_idx))
                second_mask = second_mask.to(stage2_inputs.device)

                # unroll inputs up sampling -- information need to be provided for these tokens, as they are not trustable
                unroll_inputs = torch.argmax(sampled_unroll_logits, dim=-1) # Categorical(logits=sampled_unroll_logits).sample()
                indices = indices.to(unroll_inputs.device)
                unroll_inputs[second_mask] = indices[:, :, 1][second_mask]

                # random noise injection, to provide some noise -- may just need very small noises in it
                unroll_inputs = _random_mask(unroll_inputs.clone(), self.xlmr_tok, start_random)
                start_random = start_random / 2

                features["unroll_input_ids"] = unroll_inputs
            
            all_stage2_logits.append(stage2_logits)


        stage1_loss, stage2_loss, stage3_loss = 0., 0., 0.

        with torch.backends.cudnn.flags(enabled=False):
            # print("Blank Idx >>> ", self.blank_idx)
            lprobs_raw = self.get_normalized_probs(stage1_logits, log_probs=True)
            lprobs = lprobs_raw.transpose(1, 0).contiguous()
            input_lengths = lprobs.new_full(
                (lprobs.size(1),), lprobs.size(0), dtype=torch.long
            )

            # print(labels)
            pad_mask = (labels != self.xlmr_tok.pad_token_id)
            targets_flat = labels.masked_select(pad_mask)
            target_lengths = pad_mask.sum(-1)


            stage1_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=True, # self.zero_infinity,
            )

            denoised_lprobs_raw = self.get_normalized_probs(all_stage2_logits[0], log_probs=True)
            denoised_lprobs = denoised_lprobs_raw.transpose(1, 0).contiguous()
            denoised_input_lengths = denoised_lprobs.new_full(
                (denoised_lprobs.size(1),), denoised_lprobs.size(0), dtype=torch.long
            )

            stage2_loss = F.ctc_loss(
                denoised_lprobs,
                targets_flat,
                denoised_input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=True, # self.zero_infinity,
            )

        return stage1_loss, stage2_loss


    @staticmethod
    def base_params():
        params = utils.HParams(
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=250003,
            dec_no_prefix=False,
            share_prompt=False,
            re_encoding=1
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mXLMR_SGA.base_params()


class mXLMR_SGA_iterative(modules.Module):
    def __init__(self, model, params, name="mXLMR_SGA_iterative"):
        super(mXLMR_SGA_iterative, self).__init__(name=name)
        self.params = params
        self._xlmr_model = [model.roberta]
        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        special_tokens = ["<extra_id_5>", "<extra_id_0>"]
        self.xlmr_tok.add_tokens(special_tokens)

        self.sep_id = 250001 # len(self.xlmr_tok) - 1
        self.blank_idx = self.xlmr_tok.convert_tokens_to_ids(["▁"])[0]
        params.hidden_size = model.config.hidden_size

        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.k_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self.v_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self._classifier = [model.lm_head]

        self.length_prediction = nn.Linear(model.config.hidden_size, 50)

        if params.share_prompt:
            self.prompt_model = Prompt(model, 1, params.prompt_length)
        else:
            self.prompt_model = Prompt(model, 2+params.re_encoding,
                                       params.prompt_length)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

    @property
    def xlmr_model(self):
        return self._xlmr_model[0]

    @property
    def classifier(self):
        return self._classifier[0]

    @property
    def src_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def tgt_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def softmax_embedding(self):
        return self.tgt_embedding

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])
    
    def get_normalized_probs(self, logits, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # logits = logits.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    
    def encode_semantic(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )
        
        sentence_emb = outputs.last_hidden_state[:, 0]
        sentence_cls_result = self.length_prediction(sentence_emb)

        semantic_emb = outputs.hidden_states[1:]

        # print(outputs.hidden_states[-1].requires_grad)

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]

        return past_key_values, new_sem_emb_concat, sentence_cls_result
    
    def get_translation_outputs(self, translation_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, translation_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["full_masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)
        return logits

    
    def get_length(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        # target_pos = (input_ids[0] == self.sep_id).nonzero(as_tuple=True)[0][0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )

        #############################################################
        ########### Semantic Guided Translation Prompt   ############
        #############################################################

        sentence_emb = outputs.last_hidden_state[:, 0]
        sentence_cls_result = self.length_prediction(sentence_emb)
        sentence_pred = torch.argmax(sentence_cls_result, dim=-1)

        cur_length = sentence_pred[0]

        masked_input_ids = torch.ones_like(features["masked_input_ids"]) * self.xlmr_tok.mask_token_id
        if cur_length-1 < 48:
            masked_input_ids[:, cur_length-1] = self.xlmr_tok.eos_token_id
            masked_input_ids[:, cur_length:] = self.xlmr_tok.pad_token_id
        else:
            masked_input_ids[:, -1] =  self.xlmr_tok.eos_token_id
        
        masked_input_ids[:, 0] = self.xlmr_tok.bos_token_id
        features["masked_input_ids"] = masked_input_ids

        semantic_emb = outputs.hidden_states[1:]

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[1])]
        

        return new_past_key_values


    def decode_phase1(self, features, new_past_key_values):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)

        return logits # , past_key_values, new_sem_emb_concat

    
    def denoise_phrase(self, features, past_key_values, new_sem_emb_concat):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        attention_mask = src_mask
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        new_prefix_attention_mask = torch.ones(batch_size, features["source"].shape[1]).to(input_ids.device)
        masked_attention_mask = torch.ones(batch_size, features["denoising_masked_input_ids"].shape[1]).to(input_ids.device)
        
        new_attention_mask = torch.cat((attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        # #############################################################
        # ############# Semantic Guided Denoising Prompt   ############
        # #############################################################

        final_masked_input_ids =  features["denoising_masked_input_ids"]
        input_shape = final_masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        denoising_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[2])]
        denoising_outputs = self.xlmr_model(
            input_ids=final_masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=denoising_past_key_values,
            output_hidden_states=True,
        )

        denoised_logits = self.classifier(denoising_outputs.last_hidden_state)

        return denoised_logits


    def forward(self, features, labels, denoised_labels, mode="sundae"):  
        past_key_values, new_sem_emb_concat, sentence_result = self.encode_semantic(features)
        
        raw_input_ids = features["full_masked_input_ids"] 
        loss_mask = raw_input_ids.eq(self.xlmr_tok.mask_token_id)
        loss_mask = loss_mask.float()

        tgt_length = labels.ne(self.xlmr_tok.pad_token_id).sum(-1)

        stage1_logits = self.get_translation_outputs(past_key_values[1], new_sem_emb_concat, features)
        pred_logits = torch.argmax(stage1_logits, dim=-1)
        stage1_prob = self.get_normalized_probs(stage1_logits.clone().detach(), log_probs=False)
        prob = torch.max(stage1_prob, dim=-1).values
        stage1_logits = stage1_logits[:, :labels.shape[1], :]

        stage1_logits = stage1_logits.reshape([stage1_logits.shape[0] * stage1_logits.shape[1], -1])
        stage1_loss = 0

        stage1_loss = self.criterion(stage1_logits, labels)
        stage1_loss = torch.sum(stage1_loss * loss_mask) / torch.sum(loss_mask)

        sentence_loss_fct = nn.CrossEntropyLoss()
        sentence_loss = sentence_loss_fct(sentence_result, tgt_length.view(-1))

        return stage1_loss, sentence_loss


    @staticmethod
    def base_params():
        params = utils.HParams(
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=250003,
            dec_no_prefix=False,
            share_prompt=False,
            re_encoding=1
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mXLMR_SGA_iterative.base_params()


class mXLMR_SGA_iterative_withdenoiser(modules.Module):
    def __init__(self, model, params, name="mXLMR_SGA_iterative_withdenoiser"):
        super(mXLMR_SGA_iterative_withdenoiser, self).__init__(name=name)
        self.params = params
        self._xlmr_model = [model.roberta]
        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        special_tokens = ["<extra_id_5>", "<extra_id_0>"]
        self.xlmr_tok.add_tokens(special_tokens)

        self.sep_id = 250001 # len(self.xlmr_tok) - 1
        self.blank_idx = self.xlmr_tok.convert_tokens_to_ids(["▁"])[0]
        params.hidden_size = model.config.hidden_size

        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.k_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self.v_proj = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        self._classifier = [model.lm_head]

        self.length_prediction = nn.Linear(model.config.hidden_size, 50)

        if params.share_prompt:
            self.prompt_model = Prompt(model, 1, params.prompt_length)
        else:
            self.prompt_model = Prompt(model, 2+params.re_encoding,
                                       params.prompt_length)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

    @property
    def xlmr_model(self):
        return self._xlmr_model[0]

    @property
    def classifier(self):
        return self._classifier[0]

    @property
    def src_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def tgt_embedding(self):
        return self.xlmr_model.get_input_embeddings().weight

    @property
    def softmax_embedding(self):
        return self.tgt_embedding

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])
    
    def get_normalized_probs(self, logits, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # logits = logits.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    
    def encode_semantic(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )

        sentence_emb = outputs.last_hidden_state[:, 0]
        sentence_cls_result = self.length_prediction(sentence_emb)

        semantic_emb = outputs.hidden_states[1:]

        # print(outputs.hidden_states[-1].requires_grad)

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]

        return past_key_values, new_sem_emb_concat, sentence_cls_result
    
    def get_translation_outputs(self, translation_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, translation_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["full_masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)
        return logits

    
    def get_length(self, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]


        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        past_key_values = self.prompt_model.forward(batch_size)

        #############################################################
        ################ Semantic Meaning Prompt   ##################
        #############################################################


        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.xlmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values[0],
            output_hidden_states=True,
        )

        #############################################################
        ########### Semantic Guided Translation Prompt   ############
        #############################################################

        sentence_emb = outputs.last_hidden_state[:, 0]
        sentence_cls_result = self.length_prediction(sentence_emb)
        sentence_pred = torch.argmax(sentence_cls_result, dim=-1)

        cur_length = sentence_pred[0]
        true_length = features["masked_input_ids"].ne(self.xlmr_tok.pad_token_id).sum(-1)[0]
        
        masked_input_ids = torch.ones_like(features["masked_input_ids"]) * self.xlmr_tok.mask_token_id
        if cur_length-1 < 48:
            masked_input_ids[:, cur_length-1] = self.xlmr_tok.eos_token_id
            masked_input_ids[:, cur_length:] = self.xlmr_tok.pad_token_id
        else:
            masked_input_ids[:, -1] =  self.xlmr_tok.eos_token_id
        
        masked_input_ids[:, 0] = self.xlmr_tok.bos_token_id
        features["masked_input_ids"] = masked_input_ids

        semantic_emb = outputs.hidden_states[1:]

        # print(outputs.hidden_states[-1].requires_grad)

        semantic_emb_k = [self.k_proj(item) for item in semantic_emb]
        semantic_emb_v = [self.v_proj(item) for item in semantic_emb]

        semantic_emb_k = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_k]
        semantic_emb_v = [_split_heads(item, self.prompt_model.num_heads, self.prompt_model.head_dim) for item in semantic_emb_v]

        new_sem_emb_concat = [torch.cat([item_k.unsqueeze(0), item_v.unsqueeze(0)], dim=0) for item_k, item_v in zip(semantic_emb_k, semantic_emb_v)]
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[1])]
        denoising_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[2])]

        return new_past_key_values, denoising_past_key_values


    def decode_phase1(self, features, new_past_key_values):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        input_shape = input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        batch_size = input_ids.shape[0]

        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        

        logits = self.classifier(translation_outputs.last_hidden_state)

        return logits

    
    def denoise_phase(self, features, denoising_past_key_values):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        attention_mask = src_mask
        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)
        masked_attention_mask = torch.ones(batch_size, features["denoising_masked_input_ids"].shape[1]).to(input_ids.device)
        
        new_attention_mask = torch.cat((attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        # #############################################################
        # ############# Semantic Guided Denoising Prompt   ############
        # #############################################################

        final_masked_input_ids =  features["denoising_masked_input_ids"]
        input_shape = final_masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # denoising_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(new_sem_emb_concat, past_key_values[2])]
        denoising_outputs = self.xlmr_model(
            input_ids=final_masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=denoising_past_key_values,
            output_hidden_states=True,
        )

        denoised_logits = self.classifier(denoising_outputs.last_hidden_state)

        return denoised_logits


    def get_denoise_outputs(self, denoising_prompt, semantics, features):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]

        src_mask = features["source_mask"]
        
        attention_mask = src_mask

        prefix_attention_mask = torch.ones(batch_size, self.params.prompt_length).to(input_ids.device)

        input_shape = input_ids.shape
        new_past_key_values = [torch.cat([item_past, item_sem], dim=3) for item_past, item_sem in zip(semantics, denoising_prompt)]

        new_prefix_attention_mask = torch.ones(batch_size, input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, attention_mask), dim=1)

        masked_input_ids = features["denoising_masked_input_ids"]
        masked_attention_mask = torch.ones(batch_size, masked_input_ids.shape[1]).to(input_ids.device)
        new_attention_mask = torch.cat((new_prefix_attention_mask, prefix_attention_mask, masked_attention_mask), dim=1)

        input_shape = masked_input_ids.shape

        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        translation_outputs = self.xlmr_model(
            input_ids=masked_input_ids,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            output_hidden_states=True,
        )
        
        denoising_logits = self.classifier(translation_outputs.last_hidden_state)
        return denoising_logits


    def get_glancing_logits(self, stage1_logits, labels):
        cleaned_stage1_logits = stage1_logits.clone()
        bs_size = labels.shape[0]
        
        bos = self.xlmr_tok.bos_token_id
        eos = self.xlmr_tok.eos_token_id
        pad = self.xlmr_tok.pad_token_id
        
        special_tokens_mask = labels.eq(bos) | labels.eq(eos) | labels.eq(pad)
        cleaned_stage1_logits[special_tokens_mask] = labels[special_tokens_mask]

        cleaned_stage1_logits = cleaned_stage1_logits.unsqueeze(-1)
        cleaned_stage1_logits = cleaned_stage1_logits.repeat(1, 1, 2)
        cleaned_stage1_logits = cleaned_stage1_logits.reshape(bs_size, -1)

        noised_stage1_logits = _random_mask(cleaned_stage1_logits, self.xlmr_tok, 0.2)

        return noised_stage1_logits


    def forward(self, features, labels, denoised_labels, mode="sundae"):
        mask = features["full_masked_target_mask"]
        denoised_mask = features["denoising_target_mask"]
        input_ids = features["source"]
        
        past_key_values, new_sem_emb_concat, sentence_result = self.encode_semantic(features)
        
        raw_input_ids = features["full_masked_input_ids"] 
        loss_mask = raw_input_ids.eq(self.xlmr_tok.mask_token_id)
        loss_mask = loss_mask.float()

        tgt_length = labels.ne(self.xlmr_tok.pad_token_id).sum(-1)

        stage1_logits = self.get_translation_outputs(past_key_values[1], new_sem_emb_concat, features)
        pred_logits = torch.argmax(stage1_logits, dim=-1)

        with torch.no_grad():
            glancing_logits = self.get_glancing_logits(pred_logits, labels)
            features["denoising_masked_input_ids"] = glancing_logits

        denoised_logits = self.get_denoise_outputs(past_key_values[2], new_sem_emb_concat, features)
        denoised_logits = denoised_logits[:, :labels.shape[1], :]

        # print("Denoised Logits Shape >>> ", denoised_logits.shape)

        # Stage1 loss
        stage1_prob = self.get_normalized_probs(stage1_logits.clone().detach(), log_probs=False)
        prob = torch.max(stage1_prob, dim=-1).values
        stage1_logits = stage1_logits[:, :labels.shape[1], :]

        stage1_logits = stage1_logits.reshape([stage1_logits.shape[0] * stage1_logits.shape[1], -1])
        stage1_loss = 0

        stage1_loss = self.criterion(stage1_logits, labels)
        stage1_loss = torch.sum(stage1_loss * loss_mask) / torch.sum(loss_mask)

        # Stage2 loss
        with torch.backends.cudnn.flags(enabled=False):
            pad_mask = (labels != self.xlmr_tok.pad_token_id)
            targets_flat = labels.masked_select(pad_mask)
            target_lengths = pad_mask.sum(-1)

            denoised_lprobs = self.get_normalized_probs(denoised_logits, log_probs=True)
            denoised_lprobs = denoised_lprobs.transpose(1, 0).contiguous()
            denoised_input_lengths = denoised_lprobs.new_full(
                (denoised_lprobs.size(1),), denoised_lprobs.size(0), dtype=torch.long
            )

            stage2_loss = F.ctc_loss(
                denoised_lprobs,
                targets_flat,
                denoised_input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="mean",
                zero_infinity=True, # self.zero_infinity,
            )

        sentence_loss_fct = nn.CrossEntropyLoss()
        sentence_loss = sentence_loss_fct(sentence_result, tgt_length.view(-1))

        return stage1_loss + stage2_loss, sentence_loss


    @staticmethod
    def base_params():
        params = utils.HParams(
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=250003,
            dec_no_prefix=False,
            share_prompt=False,
            re_encoding=1
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mXLMR_SGA_iterative_withdenoiser.base_params()
