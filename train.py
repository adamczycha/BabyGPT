import os
import math
from time import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from model import GPTConfig, GPT
from configparser import ConfigParser
from logger import logger
from dataloader import ChankSampler, TokenDataset
from torch.utils.data import DataLoader

config = ConfigParser()
config.read('train.cfg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx = nullcontext() if device == 'cpu' else torch.cuda.amp.autocast(device_type=device, dtype=torch.float16)


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
	try:
		assert torch.cuda.is_available()
	except AssertionError as e:
		logger.error(f'DDP only on gpu: {e}')
	dist.init_process_group(backend='nccl')
	ddp_rank = int(os.environ['RANK'])
	ddp_local_rank = int(os.environ['LOCAL_RANK'])
	ddp_world_size = int(os.environ['WORLD_SIZE'])
	device = f'cuda:{ddp_local_rank}'
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0
else:
	ddp_rank = 0
	ddp_local_rank = 0
	ddp_world_size = 1
	master_process = True
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	logger.info(f'using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
	torch.cuda.manual_seed(1337)

train_config = config['training']
model_config = config['model']
batch_size = int(train_config['batch_size'])
mini_batch = int(train_config['mini_batch'])
block_size = int(model_config['block_size'])

try:
	assert batch_size % (mini_batch * block_size * ddp_world_size) == 0
except AssertionError:
	if master_process:
		logger.critical(
			f'BATCH_SIZE is not divisible by mini_batch * block_size * world_size({mini_batch * block_size * ddp_world_size})'
		)
grad_accum_steps = batch_size // (mini_batch * block_size * ddp_world_size)
if master_process:
	logger.info(f'total desiered batch size {batch_size}')
	logger.info(f'=> calculated in gradient accumation steps: {grad_accum_steps}')


dataset = TokenDataset(config=config, split='train', seed=0)
sampler = ChankSampler(config=config, dataset=dataset, shuffle=True, seed=0)
train_loader = DataLoader(
	dataset=dataset,
	batch_size=mini_batch * block_size,
	sampler=sampler,
	pin_memory=True,
)


step_per_epoch = len(dataset) // batch_size
if master_process:
	logger.info(f' {step_per_epoch} batches in epoch')
gptconfig = GPTConfig(config)
model: DDP = GPT(gptconfig) if ddp else GPT(gptconfig)
model.to(device)
if torch.cuda.is_available():
	model.compile()
if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

opt_config = config['optimizer']
warmup_steps = int(opt_config['warmup_steps'])
max_lr = float(opt_config['max_lr'])
max_steps = int(opt_config['max_steps'])
min_lr = max_lr * float(opt_config['min_lr'])


def get_lr(it: int) -> float:
	if it < warmup_steps:
		return max_lr * (it + 1) / warmup_steps

	if it > max_steps:
		return min_lr

	decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=max_lr, device=device)

epoch = -1
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
for step in range(max_steps):
	if step % (step_per_epoch - 1) == 0:
		epoch += 1
		sampler.set_epoch(epoch)
		train_iter = iter(train_loader)
	t0 = time()
	loss_accumulation = 0.0
	ddpExist = model.no_sync() if ddp else nullcontext()
	with ddpExist:
		for micro_step in range(grad_accum_steps):
			x, y = next(train_iter)
			x, y = x.view(mini_batch, block_size), y.view(mini_batch, block_size)
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			optimizer.zero_grad()
			with ctx:
				logits, loss = model(x, y)
			loss = loss / grad_accum_steps
			loss_accumulation += loss.detach()
			if micro_step != grad_accum_steps - 1:
				scaler.scale(loss).backward()

	scaler.scale(loss).backward()
	if ddp:
		dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
	lr = get_lr(step)
	for g in optimizer.param_groups:
		g['lr'] = lr
	scaler.step(optimizer)
	scaler.update()
	if device == 'cuda':
		torch.cuda.synchronize()
	t1 = time()
	t = (t1 - t0) * 1000
	tokens_per_sec = (mini_batch * block_size * grad_accum_steps * ddp_world_size) / (t1 - t0)
	if master_process:
		logger.info(
			f'{step} loss: {loss_accumulation.item()} | iter time: {t:.2f} ms | lr: {lr:.4f} | {tokens_per_sec:.2f} tokens/sec'
		)

if ddp:
	dist.destroy_process_group()
	logger.debug('DDP destroyed sucessfuly')
