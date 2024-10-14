import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, process_rank,  num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('datasets/lalka-tom-pierwszy.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        self.current_place = B*T*process_rank
        print(f'Dataset has {len(text)} characters and {len(self.tokens)} tokens')
        print(f'Number of batches {len(self.tokens)//(B*T)} ')

    def get_batch(self):
        if (self.current_place + self.B*self.T+1) >= len(self.tokens):
            self.current_place = self.B*self.T*self.process_rank
        x = self.tokens[self.current_place:self.current_place+(self.B*self.T)].view(self.B,self.T)
        y = self.tokens[self.current_place + 1:self.current_place+(self.B*self.T) + 1].view(self.B,self.T)
        self.current_place += self.B* self.T * self.num_processes
        return x, y
