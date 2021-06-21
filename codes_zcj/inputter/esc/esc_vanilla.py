# coding=utf-8

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputter.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader


class ESCInputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features
        
        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader
        
        # valid
        self.valid_dataloader = DynamicBatchingLoader
        
        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
        self,
        src_input_ids,
        tgt_input_ids, tgt_label_ids,
    ):
        self.src_input_ids = src_input_ids
        self.src_len = len(src_input_ids)
        
        self.tgt_input_ids = tgt_input_ids
        self.tgt_len = len(tgt_input_ids)
        self.tgt_label_ids = tgt_label_ids

        self.input_len = self.src_len + self.tgt_len


def featurize(
    bos, eos, vocab_size,
    context, max_src_len, src_strat_ids,
    response, max_tgt_len,
):
    assert len(context) == len(src_strat_ids)
    src_input_ids = [[min(st, vocab_size + 1)] + ctx + [eos] for ctx, st in zip(context, src_strat_ids)]
    src_input_ids = sum(src_input_ids, [])[:-1]
    src_input_ids = src_input_ids[-max_src_len:]
    
    tgt_label_ids = (response + [eos])[:max_tgt_len]
    tgt_input_ids = [bos] + tgt_label_ids[:-1]
    
    assert len(tgt_input_ids) == len(tgt_label_ids), tgt_input_ids[1:] == tgt_label_ids[:-1]

    return InputFeatures(
        src_input_ids,
        tgt_input_ids, tgt_label_ids,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    dialog = data['dialog']

    assert len(toker) == toker.vocab_size + 1 + 1
    
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(_norm(x)))
    inputs = []
    context = []
    strat_ids = []
    
    for i in range(len(dialog)):
        text = process(dialog[i]['text'])
        if dialog[i]['speaker'] == 'usr':
            strat_id = toker.vocab_size
        else:
            strat_id = toker.vocab_size + 1
        
        if i > 0 and dialog[i]['speaker'] == 'sys':
            res = {
                'context': context.copy(),
                'src_strat_ids': strat_ids.copy(),
                
                'response': text,
            }
            
            inputs.append(res)
        
        context = context + [text]
        strat_ids = strat_ids + [strat_id]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []
    
    assert 'max_src_len' in kwargs, 'you should give max_src_len'
    max_src_len = kwargs.get('max_src_len')
    assert 'max_tgt_len' in kwargs and kwargs.get('max_tgt_len') is not None, 'you should give max_tgt_len'
    max_tgt_len = kwargs.get('max_tgt_len')
    
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos, toker.vocab_size,
            ipt['context'], max_src_len, ipt['src_strat_ids'],
            ipt['response'], max_tgt_len,
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        assert len(toker) == toker.vocab_size + 1 + 1
        
        src_input_ids = pad_sequence([torch.tensor(f.src_input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        src_attention_mask = pad_sequence([torch.tensor([1.] * f.src_len, dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)
        src_len = torch.tensor([f.src_len for f in features], dtype=torch.long)
        
        tgt_input_ids = pad_sequence([torch.tensor(f.tgt_input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        tgt_label_ids = pad_sequence([torch.tensor(f.tgt_label_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=-1)
        tgt_len = torch.tensor([f.tgt_len for f in features], dtype=torch.long)
        
        res = {
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'src_len': src_len,
            
            'tgt_input_ids': tgt_input_ids,
            'tgt_label_ids': tgt_label_ids,
            'tgt_len': tgt_len,
        }
        
        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()
            
            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []
                    
            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch
                
        except StopIteration:
            pass
    
    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker):
    res = FeatureDataset.collate(features, toker)
    
    res['tgt_input_ids'] = res.pop('tgt_input_ids')[:, :1]
    res.pop('tgt_label_ids')
    
    res['batch_size'] = res['src_input_ids'].size(0)

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give max_src_len'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    
    features = []
    turn_lens = []
    posts = []
    references = []
    for line in tqdm.tqdm(reader, total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        features.extend(convert_inputs_to_features(inputs, toker, **kwargs))
        turn_lens.append(len(inputs))
        for i in range(len(inputs)):
            ipt = inputs[i]

            src_strat_ids = ipt['src_strat_ids'][::-1]
            ctx_idx = 0
            while ctx_idx < len(src_strat_ids) and src_strat_ids[ctx_idx] == toker.vocab_size:
                ctx_idx += 1
            context = ipt['context'][::-1][:ctx_idx][::-1]

            posts.append(toker.decode([ee for e in context for ee in e + [toker.eos_token_id]]))
            references.append(toker.decode(ipt['response']))
    
        if len(turn_lens) == infer_batch_size:
            yield prepare_infer_batch(features, toker), posts, references, turn_lens
            features = []
            turn_lens = []
            posts = []
            references = []

    if len(turn_lens) > 0:
        yield prepare_infer_batch(features, toker), posts, references, turn_lens
