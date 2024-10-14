import os
import math
from time import time
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from model import GPTConfig, GPT
from dataloader import DataLoader


#basic training config:
warmup_steps = 10
max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1
batch_size = 512   # in tokens
mini_batch = 4  #real batch size used to simulate bigger batches

#model configuration:
block_size: int = 128
vocab_size: int = 50257
n_head: int = 8
n_layer: int = 8
n_embd: int  = 128

#system
device = 'cpu'




device = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=torch.float16)


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), 'DDP have measurable effect only on GPU'
    dist.init_process_group(backend='nccl')
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

assert batch_size % (mini_batch * block_size * ddp_world_size) == 0, f'BATCH_SIZE is not devidable by B*T({mini_batch * block_size * ddp_world_size})'
grad_accum_steps = batch_size // (mini_batch * block_size * ddp_world_size)
if master_process:
    print(f'total desiered batch size {batch_size}')
    print(f'=> calculated in gradient accumation steps: {grad_accum_steps}') 

train_loader = DataLoader(B=mini_batch, T=block_size, process_rank=ddp_rank, num_processes= ddp_world_size)

gptconfig = GPTConfig(block_size = block_size, vocab_size= vocab_size, n_head = n_head, n_layer = n_layer, n_embd = n_embd)
model = GPT(gptconfig)
model.to(device)
if torch.cuda.is_available():
    model.compile()
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizer(weight_decay = 0.1, learning_rate = max_lr, device = device)

scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
for step in range(max_steps):
    t0 = time()
    loss_accumulation = 0.0
    ddpExist = model.no_sync() if ddp else nullcontext() 
    with ddpExist:
        for micro_step in range(grad_accum_steps):
            x , y = train_loader.get_batch()
            x , y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with ctx:
                logits, loss = model(x,y)
            loss = loss / grad_accum_steps
            loss_accumulation += loss.detach()
            scaler.scale(loss).backward()
    if ddp:
        scaler.scale(loss).backward()
        dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm =1.0 )
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time()
    t = (t1-t0)*1000
    tokens_per_sec = (train_loader.T*train_loader.B * grad_accum_steps * ddp_world_size)/(t1-t0)
    if  master_process:
        print(f'{step} loss: {loss_accumulation.item()} | iter time: {t:.2f} ms | lr: {lr:.4f} | {tokens_per_sec:.2f} tokens/sec')

    if ddp:
        dist.destroy_process_group()








