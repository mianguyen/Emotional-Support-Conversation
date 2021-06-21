# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.base_framework import BaseFramework
from framework.generation_utils import top_k_top_p_filtering
from transformers.tokenization_utils import PreTrainedTokenizer
from .PARAMS import SAMPLE, TEMPERATURE


class ESC(BaseFramework):
    def __init__(self, encoder=None, decoder=None, toker=None, **kwargs):
        super().__init__(decoder.config)
        self.encoder = encoder
        self.decoder = decoder
        self.toker: PreTrainedTokenizer = toker
        
        assert self.encoder is not None and self.decoder is not None
        
        assert hasattr(self.decoder.config, 'expand_vocab') and self.decoder.config.expand_vocab
        assert hasattr(self.decoder.config, 'expand_vocab_size')
        assert len(self.toker) == self.toker.vocab_size + self.decoder.config.expand_vocab_size
        self.expand_vocab_size = self.decoder.config.expand_vocab_size
        assert self.expand_vocab_size == 2
        
        self.encoder.resize_token_embeddings(len(self.toker))
        self.decoder.resize_token_embeddings(len(self.toker))
        
        assert kwargs.get('share_word_embeddings', False)
        encoder_embeddings = self.encoder.get_word_embeddings()
        decoder_embeddings = self.decoder.get_word_embeddings()
        assert decoder_embeddings.weight.size() == encoder_embeddings.weight.size(), (
            f'encoder and decoder cannot share embeddings with different shapes:'
            f'encoder {encoder_embeddings.weight.size()} vs. decoder {decoder_embeddings.weight.size()}'
        )
        decoder_embeddings.weight = encoder_embeddings.weight
        
        self.decoder.tie_weights()
    
    def encode(
        self,
        src_input_ids,
        src_attention_mask,
        encoded_info,
    ):
        assert src_input_ids.dim() == 2

        hidden_states, *_ = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
        )
        res = {
            'encoder_hidden_states': hidden_states,
            'encoder_attention_mask': src_attention_mask,
        }
        encoded_info.update(res)
    
    def forward(
        self,
        src_input_ids,
        src_attention_mask,
        tgt_input_ids,
        tgt_label_ids,
        pointwise=False,
        **kwargs,
    ):
        encoded_info = kwargs
        self.encode(src_input_ids, src_attention_mask, encoded_info=encoded_info)
        
        outputs = self.decoder(
            input_ids=tgt_input_ids,
            **encoded_info
        )
        lm_logits = outputs[0]
        
        if pointwise:
            assert not self.training
            lm_logits = lm_logits[..., :-self.expand_vocab_size].contiguous()
        
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                               ignore_index=-1, reduction='none')
        loss = loss.view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(loss)
        loss_value = torch.sum(loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        
        if not pointwise:
            res = {
                'all': loss_value,
                'ppl': ppl_value,
            }
            return res
        else:
            return loss, label_size
    
    @torch.no_grad()
    def generate(
        self,
        src_input_ids,
        src_attention_mask,
        tgt_input_ids,
        **kwargs
    ):
        assert tgt_input_ids.size(1) == 1
        
        encoded_info = kwargs
        self.encode(src_input_ids, src_attention_mask, encoded_info=encoded_info)

        #encoded_info.update({'src_input_ids': src_input_ids,})
        encoded_info.update({'expand_vocab_size': self.expand_vocab_size})
        
        return encoded_info, super().generate(input_ids=tgt_input_ids, **encoded_info)
