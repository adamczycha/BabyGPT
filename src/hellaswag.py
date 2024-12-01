import requests
import os
from tqdm import tqdm
import tiktoken
import torch
import json

PATH = 'benchmarks/hellaswag'

def download_val():
    response = requests.get('https://raw.githubusercontent.com/rowanz/hellaswag/refs/heads/master/data/hellaswag_val.jsonl', stream=True)
    os.makedirs('../benchmarks/hellaswag', exist_ok=True)
    data_size = int(response.headers.get('content-length',0))
    cumulative_download = 0
    with open(f'{PATH}/hellaswag_val.jsonl', mode='wb') as file, tqdm(
        desc='hellaswag_val',
        total=data_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
           size = file.write(data)
           cumulative_download += size
           if cumulative_download > bar.total:
               bar.total = cumulative_download
           bar.update(size) 
        
def iterate_examples():
    with open('benchmarks/hellaswag/hellaswag_val.jsonl', 'rb') as file:
        for line in file:
            example = json.loads(line)
            yield example

def prepare_example(example):
    ctx = example['ctx']
    endings = example['endings']
    label = example['label']
    
    enc = tiktoken.get_encoding('gpt2')
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(' '+ end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    tokens = torch.zeros([4,1024], dtype=torch.long)
    mask = torch.zeros([4,1024], dtype=torch.long)
    for i,(tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i,:len(tok_row)] = torch.tensor([tok_row]) 
        mask[i,:len(mask_row)] = torch.tensor([mask_row])
    
    
    return tokens, mask, label

download_val()