
from dataclasses import dataclass
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
import math
import pandas as pd
import numpy as np
from time import time
import inspect
import tiktoken
import os



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_layer: int = 12
    n_embd: int  = 768
    


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

#         att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = att.masked_fill(self.bias[:,:,:T,:T] == 0, (-np.inf))
#         att = F.softmax(att, dim = -1)
#         y = att @ v # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
    
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean= 0.0, std= std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Model cannot operate {T} as a block size maximum is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        idx = pos_emb + tok_emb
        for block in self.transformer.h:
            idx = block(idx)
        idx = self.transformer.ln_f(idx)
        logits = self.lm_head(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 




    @classmethod
    def from_pretrained(cls, model_type ):
        assert model_type in {'gpt2', 'gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
        print(f'loading weights from pretrained gpt: {model_type}' )
        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd =768),
            'gp2-medium':   dict(n_layer = 24, n_head = 16, n_embd =1024),
            'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embd =1280),
            'gpt2-xl':      dict(n_layer = 48, n_head = 25, n_embd =1600),
        }[model_type]
        config_args['vocab_size']= 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate = 6e-4, device = device):
        param_dict = {name: param for name, param in self.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}
        
        decay_params = [param for param  in param_dict.values() if param.dim() >= 2]
        nodecay_params = [param for param in param_dict.values() if param.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decayed_params = sum(param.numel() for param in decay_params)
        num_nodecayed_params = sum(param.numel() for param in nodecay_params)
        print(f'num decayed parameter tensors {len(decay_params)}, with {num_decayed_params} weights')
        print(f'num no decayed paramter tensors {len(nodecay_params)}, with {num_nodecayed_params} weights')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(model.parameters(),betas = (0.9, 0.95), eps = 10e-8,lr = learning_rate, fused = use_fused)
        return optimizer

class DataLoader:
    def __init__(self, B, T, process_rank,  num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('/kaggle/input/lalka/lalka-tom-pierwszy.txt', 'r', encoding='utf-8') as f:
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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), 'DDP have measurable effect only on GPU'
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'gpu' if torch.cuda.is_available() else 'cuda'
    print(f'using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

BATCH_SIZE = 524288
B = 4
T = 1024
assert BATCH_SIZE % (B*T) == 0, f'BATCH_SIZE is not devidable by B*T({B*T})'
grad_accum_steps = BATCH_SIZE // (B*T)
if master_process:
    print(f'total desiered batch size {BATCH_SIZE}')
    print(f'=> calculated in gradient accumation steps: {grad_accum_steps}') 



train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes= ddp_world_size)


model = GPT(GPTConfig(vocab_size =50304, block_size=T))
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

warmup_steps = 10
max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizer(weight_decay = 0.1, learning_rate = max_lr)
scaler = torch.cuda.amp.GradScaler()
for step in range(max_steps):
    t0 = time()
    loss_accumulation = 0.0
    for micro_step in range(grad_accum_steps):
        x , y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accumulation += loss.detach() 
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm =1.0 )
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    t1 = time()
    t = (t1-t0)*1000
    tokens_per_sec = (train_loader.T*train_loader.B * grad_accum_steps * ddp_world_size)/(t1-t0)
    if  master_process:
        print(f'{step} loss: {loss_accumulation.item()} | iter time: {t:.2f} ms | lr: {lr:.4f} | {tokens_per_sec:.2f} tokens/sec')

    if ddp:
        dist.destroy_process_group()








