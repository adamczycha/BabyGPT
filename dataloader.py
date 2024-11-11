import numpy as np
import os 
import random
import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from pathlib import Path
from configparser import ConfigParser
import math
from logger import logger 



document_split_dtype = np.dtype([
                ('first', np.int64),
                ('second', np.int64)
            ])


#sampler should be DDP compatible and multi epoch compatible


class ChankSampler(Sampler):
    def __init__(self, config,  dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.chank_size = dataset.chank_size
        self.seed = seed
        self.shuffle = shuffle
        self.config = config
        self.epoch = 0
        


    def __iter__(self):
        self.chank_indices = []
        ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        for i in range(len(self.dataset)//self.chank_size):
            start = i* self.chank_size
            self.chank_indices.append((start, start + (self.chank_size-1)))

        mini_batch = int(self.config['training']['mini_batch'])
        block_size = int(self.config['model']['block_size'])
        closes_fiting_multiple = math.floor(len(self.dataset)/ddp_world_size/mini_batch/block_size)
        closes_fiting_size = closes_fiting_multiple * ddp_world_size * mini_batch * block_size
        if int(os.environ.get('RANK', 1)) == 0:
            logger.info(f'Drop last {len(self.dataset) - closes_fiting_size} tokens')

        for i in range(len(self.chank_indices)-1, 0, -1):
            if self.chank_indices[i][0] < closes_fiting_size:
                self.chank_indices[i] = (self.chank_indices[i][0], closes_fiting_multiple)
                self.chank_indices = self.chank_indices[:i+1]
                break

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(self.chank_indices)

        if ddp_world_size > 1:
            ddp_rank = int(os.environ.get('RANK', 0))
            for chank_start, chank_end in self.chank_indices:
                indices = list(range(chank_start , chank_end))
                for idx in indices[ddp_rank*block_size*mini_batch:(1+ddp_rank)*block_size*mini_batch]:
                    yield idx
        else:
            for chank_start, chank_end in self.chank_indices:
                for idx in range(chank_start , chank_end):
                    yield idx
            
    def set_epoch(self, epoch):
        self.epoch = epoch

class TokenDataset(Dataset):
    def __init__(self, config: ConfigParser,  split: str , seed: int) -> None:
        # dataset path: {dataset}/test/test.bin dtype = int16
        self.seed = seed 
        self.config = config
        block_size = int(config['model']['block_size'])
        mini_batch = int(config['training']['mini_batch'])
        # including <|endoftext|> at the end of each chunk
        self.chank_size = os.environ.get('WORLD_SIZE', 1) * mini_batch * block_size if config['data'].get('chank_size') is None else int(config['data']['chank_size'])  
        dataset = config['data']['dataset']

        self.tokens = np.memmap(f'{dataset}/{split}/{split}.bin', dtype = np.uint16,  mode = 'r')
        
    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            #chank size -1 last index of the chank is needed not length 
            start_chank_index = idx.start // (self.chank_size -1 ) 
            end_chank_index = idx.stop // (self.chank_size -1)
            if start_chank_index == end_chank_index:
                return self.tokens[(idx.start - start_chank_index) : (idx.stop - end_chank_index)]
            else:
                token_seq = []

                for chank_index in range(start_chank_index, end_chank_index +1):
                    start = chank_index * self.chank_size
                    end = start + (self.chank_size -1) #space for EOF
            
                    if chank_index == start_chank_index:
                        token_seq.extend(self.tokens[(idx.start - chank_index):(end - chank_index)])
                        token_seq.append(50256)
                    elif chank_index == end_chank_index:
                         token_seq.extend(self.tokens[(start - chank_index):(idx.stop - chank_index)])
                    else:
                        token_seq.extend(self.tokens[start:end])
                        token_seq.append(50256)
            return token_seq
        else:
            chank_index = idx//(self.chank_size -1)            
            return 50256 if idx % (self.chank_size -1) == 0 and idx != 0 else self.tokens[idx - chank_index]



