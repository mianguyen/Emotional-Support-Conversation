import json
import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
random.seed(13)


def _norm(x):
    return ' '.join(x.strip().split())


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
original = json.load(open('./corpus.json'))

def process_data(d):
    emotion = d['emotion']
    situation = d['complaint']

    d = d['contents']
    dial = []
    for uttr in d:
        text = _norm(uttr['content'])
        role = uttr['role']
        if role == 'speaker':
            dial.append({
                'text': text,
                'speaker': 'usr',
                'segment_id': 0,
            })
        else:
            dial.append({
                'text': text,
                'speaker': 'sys',
                'segment_id': 1,
                'strategy': strat2id[uttr['payload']['method']],
            })
    res = {
        'emotion': emotion,
        'situation': situation,
        'dialog': dial,
    }
    return res

data = []

with mp.Pool(processes=mp.cpu_count()) as pool:
    for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)

random.shuffle(data)
dev_size = int(0.15 * len(data))
test_size = int(0.15 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./train.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')
with open('./sample.json', 'w') as f:
    json.dump(train[:10], f, ensure_ascii=False, indent=2)

print('valid', len(valid))
with open('./valid.txt', 'w') as f:
    for e in valid:
        f.write(json.dumps(e) + '\n')

print('test', len(test))
with open('./test.txt', 'w') as f:
    for e in test:
        f.write(json.dumps(e) + '\n')
