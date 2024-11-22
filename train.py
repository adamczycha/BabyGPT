import os
import math
from time import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from src.model import GPT
from src.scheduler import CosineScheduler
from src.logger import logger
from src.dataset import  TokenDataset
from src.sampler import ChankSampler
from src.dataloader import custom_collate_fn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml


with open('train_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=torch.float16)


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

resume_run = False
if config['training']['init_from'] == 'resume':
	resume_run = True
	model_dir = Path(config['training']['path_to_resume_training'])
	assert os.path.exists(model_dir)

	checkpoint = torch.load(model_dir)
	config['model']= checkpoint['config']['model']
	model = GPT(config)
	model.load_state_dict(checkpoint['model'])
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler = CosineScheduler(optimizer, config)
	scheduler.load_state_dict(checkpoint['scheduler'])
	logger.info(f'Continue traning. Starting on {checkpoint["step"]} step with loss {checkpoint["loss"]}')

elif config['training']['init_from'] == 'gpt2':
	model = GPT.from_pretrained('gpt2')
	config[model] = model.config
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	scheduler = CosineScheduler(optimizer, config) 
	logger.info(f'GPT from pretrained loaded.')

elif config['training']['init_from'] == 'scratch':

	model = GPT(config)
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	scheduler = CosineScheduler(optimizer, config) # does not depend on optimizer learnig rate



train_config = config['training']
model_config = config['model']
batch_size = train_config['batch_size']
mini_batch = train_config['mini_batch']
block_size = model_config['block_size']


try:
	assert batch_size % (mini_batch * block_size * ddp_world_size) == 0
except AssertionError:
	if master_process:
		logger.critical(f'BATCH_SIZE is not divisible by mini_batch * block_size * world_size({mini_batch * block_size * ddp_world_size})')
grad_accum_steps = batch_size // (mini_batch * block_size * ddp_world_size)
if master_process:
	logger.info(f'total desiered batch size {batch_size}')
	logger.info(f'=> calculated in gradient accumation steps: {grad_accum_steps}')


if torch.cuda.is_available() and bool(train_config['compile']):
	model.compile()

dataset = TokenDataset(config=config, split='train', seed=0)
sampler = ChankSampler(config=config, dataset=dataset, shuffle=True, seed=0)
train_loader = DataLoader(dataset=dataset, batch_size=(mini_batch *block_size),  sampler=sampler,  collate_fn=custom_collate_fn, pin_memory=True)

if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])

step_per_epoch = len(dataset) // batch_size
if master_process:
	logger.info(f' {step_per_epoch} batches in epoch')

epoch = -1 if 'checkpoint' not in locals() else checkpoint['epoch']   
last_step = 0 if 'checkpoint' not in locals() else checkpoint['step'] + 1
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
for step in range(last_step, int(config['optimizer']['max_steps'])):
	t0 = time()
	loss_accumulation = 0.0

	if step % (step_per_epoch - 1) == 0 or resume_run:
		epoch += 1
		sampler.set_epoch(epoch )
		train_iter = iter(train_loader)
		resume_run = False
	ddpExist = model.no_sync() if ddp else nullcontext()
	with ddpExist:
		for micro_step in range(grad_accum_steps):
			x, y = next(train_iter)
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
	scaler.step(optimizer)
	scheduler.step()
	scaler.update()
	if device == 'cuda':
		torch.cuda.synchronize()
	
	if master_process:
		tokens_per_sec = (mini_batch * block_size * grad_accum_steps * ddp_world_size) / (time() - t0)
		logger.info(f'{step} loss: {loss_accumulation.item()} | iter time: {(time() - t0) * 1000:.2f} ms | lr: {scheduler.get_lr()[0]:.4f} | {tokens_per_sec:.2f} tokens/sec')
		saving_config = config['saving']
		#saving
		if saving_config['save_checkpoints'] == True and step % saving_config['save_every_n_batches'] == 0:
			state = {
					'step': step,
					'epoch': epoch,
					'config': config,
					'model': model.state_dict(),
					'loss': loss_accumulation.item() 
					}
			if saving_config['save_with_resume_option'] == True:
				state['optimizer'] =  optimizer.state_dict()
				state['scheduler'] =  scheduler.state_dict() 
			torch.save(state, f'checkpoint_{step/grad_accum_steps/step_per_epoch:.4f}.pth') 
			logger.info(f'Checkpoint saved. On {step/grad_accum_steps/step_per_epoch:.4f} epoch. Resumeable save: {str(True) if bool(saving_config["save_with_resume_option"]) == True else str(False)}')





if ddp:
	dist.destroy_process_group()
	logger.debug('DDP destroyed sucessfuly')
