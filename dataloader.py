import numpy as np
import os 
import random
import os
import pandas as pd
from torch.utils.data import Dataset
import torch.distributed as dist
from pathlib import Path
from configparser import ConfigParser

document_split_dtype = np.dtype([
                ('first', np.int64),
                ('second', np.int64)
            ])

class TokenDataset(Dataset):
    def __init__(self, config: ConfigParser, split: str , seed: int) -> None:
        # dataset path: {dataset}/test/test.bin dtype = int16
        self.B = int(config['training']['mini_batch'])
        self.T = int(config['model']['block_size'])
        self.process_rank = os.environ.get('RANK', 0)
        self.num_processes = os.environ.get('WORLD_SIZE', 1)
        self.seed = seed 
        self.epoch = 0
        self.processed_tokens = 0
        self.chank_size = 20000
        dataset = config['data']['dataset']
        cache_dir = Path(f'{dataset}/.cache/{split}')
        if self.process_rank == 0:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokens = np.memmap(f'{dataset}/{split}/{split}.bin', dtype = np.uint16,  mode = 'r')
        
        # shuffle data chankes
        if self.process_rank == 0:
            # chank data every 20k tokens and shuffle the chankes add <|endoftext|> between every chank 
            
            document_beg_end_indices = []
            for i in range(len(self.tokens)//self.chank_size):
                document_beg_end_indices.append((i*self.chank_size,(i+1)*self.chank_size))

            self.document_split  = np.memmap(f'{dataset}/.cache/{split}/{split}.bin', dtype=document_split_dtype, mode = 'w+', shape=((len(document_beg_end_indices),)))
            self.document_split[: len(document_beg_end_indices)] = document_beg_end_indices
            self.document_split.flush()
            if self.num_processes > 1:
                dist.barrier()
            if self.process_rank != 0:
                 self.document_split = np.memmap(f'{dataset}/.cache/{split}/{split}.bin', dtype=document_split_dtype, mode = 'r+')

    def shuffle(self, epoch):
        random.seed(self.seed + epoch)
        random.shuffle(self.document_split)


    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start_chank_index = idx.start // self.chank_size 
            start_token_index = idx.start % self.chank_size
            end_chank_index = idx.stop // self.chank_size
            end_token_index = idx.stop % self.chank_size - (end_chank_index - start_chank_index) # space for <|endoftext|>
            if start_chank_index == end_chank_index:
                return self.tokens[(self.document_split[start_chank_index][0] + start_token_index):(self.document_split[end_chank_index][0] + end_token_index)]
            else:
                token_seq = []
                for chank_index in range(start_chank_index, end_chank_index):
                    if chank_index == start_chank_index:
                        start, end = self.document_split[start_chank_index]
                        token_seq.extend(self.tokens[(start+start_token_index):(end)])
                    elif chank_index == end_chank_index:
                        start, end = self.document_split[end_chank_index]
                        token_seq.extend(self.tokens[start:(end+end_token_index)])
                    else:
                        start, end = self.document_split[chank_index]
                        token_seq.extend(self.tokens[start:end])
                    token_seq.extend([50256])
            return token_seq
        chank_index = idx//self.chank_size
        token_index = idx % self.chank_size
        return self.tokens[(self.document_split[chank_index][0] + token_index)]



