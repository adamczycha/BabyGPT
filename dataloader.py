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


class ChankSampler(Sampler):
    def __init__(self, config,  dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.chank_size = dataset.chank_size
        self.seed = seed
        self.shuffle = shuffle
        self.config = config
        self.epoch = 0
    
    def __iter__(self):
        self.document_indices = []
        ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        self.document_indices = self.find_document_boundries()

        # drop last to fit ideally for multiple GPU run 
        mini_batch = int(self.config['training']['mini_batch'])
        block_size = int(self.config['model']['block_size'])
        closes_fiting_multiple = math.floor(len(self.dataset)/ddp_world_size/mini_batch/block_size)
        closes_fiting_size = closes_fiting_multiple * ddp_world_size * mini_batch * block_size
        if int(os.environ.get('RANK', 1)) == 0:
            logger.info(f'Drop last {len(self.dataset) - closes_fiting_size} tokens')

        for i in range(len(self.document_indices)-1, 0, -1):
            if self.document_indices[i][0] < closes_fiting_size:
                self.document_indices[i] = (self.document_indices[i][0], closes_fiting_multiple-1) # space for EOF at the end 
                self.document_indices = self.document_indices[:i+1]
                break

        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(self.document_indices)


        if ddp_world_size > 1:
            ddp_rank = int(os.environ.get('RANK', 0))
            for chank_start, chank_end in self.document_indices[ddp_rank::ddp_world_size]:
                for idx in range(chank_start , chank_end):
                    yield idx
                
        else:
            for chank_start, chank_end in self.document_indices:
                for idx in range(chank_start , chank_end): 
                    yield idx

    def find_document_boundries(self):
        document_boundry = []
        cursor = 30000000
        step_size = 30000000
        search_range = 20000

        while cursor < len(self.dataset):
            # Search for EOF token in a fixed range around the cursor\
            EOF_list = []
            while EOF_list == []:
                search_start = max(cursor - search_range, 0)
                search_end = min(cursor + search_range, len(self.dataset))
                EOF_list = np.where(self.dataset.tokens[search_start:search_end] == 50256)[0]
                
                if EOF_list.size == 0:
                    search_start += search_range
                    search_end += search_end
                else:
                    EOF_index = search_start + EOF_list[0]
                    document_boundry.append(EOF_index)
                    cursor = EOF_index + step_size
                    break
            cursor += step_size

        # Add the start and end boundaries
        document_boundry.insert(0, 0)
        document_boundry.append(len(self.dataset))

        # Create document indices
        return  [(document_boundry[i], document_boundry[i + 1]) for i in range(len(document_boundry) - 1)]
                
    def set_epoch(self, epoch):
        self.epoch = epoch

class TokenDataset(Dataset):
    def __init__(self, config: ConfigParser,  split: str , seed: int) -> None:
        # dataset path: {dataset}/test/test.bin dtype = int16
        self.seed = seed 
        self.config = config
        block_size = int(config['model']['block_size'])
        mini_batch = int(config['training']['mini_batch'])
        ddp_world_size = os.environ.get('WORLD_SIZE', 1)
        # including <|endoftext|> at the end of each chunk
        self.chank_size = ddp_world_size * mini_batch * block_size if config['data'].get('chank_size') is None else int(config['data']['chank_size'])  
        dataset = config['data']['dataset']

        self.tokens = np.memmap(f'{dataset}/{split}/{split}.bin', dtype = np.uint16,  mode = 'r')
        closes_fiting_multiple = math.floor(len(self.tokens)/ddp_world_size/mini_batch/block_size)
        self.closes_fiting_size = closes_fiting_multiple * ddp_world_size * mini_batch * block_size
        
    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return torch.tensor(self.tokens[idx.start : idx.stop], dtype= torch.float32)
        else: 
            if idx == self.closes_fiting_size -1 : # change last index of data to EOF and its target to EOF, theoreticly -2nd token's target should be changed also to EOF but it is not natural ending of the sentense. Therefor I will allow prediction and change only last token . 
                return torch.tensor(self.tokens[50265], dtype= torch.float32), torch.tensor(self.tokens[50265], dtype= torch.float32)
            else:
                return torch.tensor(self.tokens[idx], dtype= torch.float32), torch.tensor(self.tokens[idx+1], dtype= torch.float32)



