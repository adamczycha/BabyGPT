import os
from time import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
from src.model import GPT
from src.scheduler import CosineScheduler
from src.logger import logger
from src.dataset import TokenDataset
from src.sampler import ChankSampler
from src.dataloader import custom_collate_fn
from torch.utils.data import DataLoader
from pathlib import Path
from src.hellaswag import HellaSwag
import yaml
from transformers import AutoTokenizer
import torch.nn.functional as F


with open('train_config.yaml', 'r') as file:
	config = yaml.safe_load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ptdtype)


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
if config['general']['init_from'] == 'resume':
	resume_run = True
	model_dir = Path(config['general']['path_to_resume_training'])
	assert os.path.exists(model_dir), 'You want to resume training, but file does not exist!'

	checkpoint = torch.load(model_dir)
	config['model'] = checkpoint['config']['model']
	model = GPT(config)
	model.load_state_dict(checkpoint['model'])
	model.to(device)
	if config['general']['train']:
		optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler = CosineScheduler(optimizer, config)
		scheduler.load_state_dict(checkpoint['scheduler'])
		if master_process:
			logger.info(f'Continue traning. Starting on {checkpoint["step"]} step with loss {checkpoint["loss"]}')

elif config['general']['init_from'] == 'gpt2':
	model = GPT.from_pretrained('gpt2')
	config[model] = model.config
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	scheduler = CosineScheduler(optimizer, config)
	if master_process:
		logger.info('GPT from pretrained loaded.')

elif config['general']['init_from'] == 'scratch':
	model = GPT(config)
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	scheduler = CosineScheduler(optimizer, config)  # does not depend on optimizer learnig rate

batch_size = config['training']['batch_size']
mini_batch = config['training']['mini_batch']
block_size = config['model']['block_size']

if config['general']['train']:
	try:
		assert batch_size % (mini_batch * block_size * ddp_world_size) == 0
	except AssertionError:
		if master_process:
			logger.critical(f'BATCH_SIZE is not divisible by mini_batch * block_size * world_size. Batch_size => ({mini_batch * block_size * ddp_world_size})')
	grad_accum_steps = batch_size // (mini_batch * block_size * ddp_world_size)
	if master_process:
		logger.info(f'total desiered batch size {batch_size}')
		logger.info(f'=> calculated in gradient accumation steps: {grad_accum_steps}')

if torch.cuda.is_available() and config['training']['compile'] and config['general']['train']:
	model.compile()

if config['general']['train']:
	train_dataset = TokenDataset(config=config, split='train')
	sampler = ChankSampler(config=config, dataset=train_dataset, shuffle=True, seed=0)
	train_loader = DataLoader(
		dataset=train_dataset, batch_size=(mini_batch * block_size), sampler=sampler, collate_fn=custom_collate_fn, pin_memory=True
	)
	step_per_epoch = len(train_dataset) // batch_size
	if master_process:
		logger.info(f' {step_per_epoch} batches in epoch')

if config['general']['val']:
	val_dataset = TokenDataset(config=config, split='val')
	val_sampler = ChankSampler(config, dataset=val_dataset, split='validation')
	val_loader = DataLoader(
		dataset=val_dataset, batch_size=(mini_batch * block_size), sampler=val_sampler, collate_fn=custom_collate_fn, pin_memory=True
	)
	
if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])



epoch = -1 if ('checkpoint' not in locals())  or not config['general']['train'] else checkpoint['epoch']
last_step = 0 if ('checkpoint' not in locals())  or not config['general']['train'] else checkpoint['step'] + 1
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

for step in range(last_step, int(config['optimizer']['max_steps'])):
	if config['general']['train'] and (step % (step_per_epoch - 1) == 0 or resume_run):
		epoch += 1
		sampler.set_epoch(epoch)
		train_iter = iter(train_loader)
		resume_run = False

	with (model.no_sync() if ddp else nullcontext()):
		if config['general']['sample'] and (((step % config['sampling']['sample_every_n'] == 0) or step == config['optimizer']['max_steps']-1)  or config['sampling']['force_sample']):
			model.eval()
			tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, clean_up_tokenization_spaces = False)
			tokens = tokenizer.encode(config['sampling']['starting_sequence'])
			tokens = torch.tensor(tokens, dtype=torch.long)
			tokens = tokens.unsqueeze(0).repeat(config['sampling']['samples_per_rank'], 1)
			xgen = tokens.to(device)
			sample_rng = torch.Generator(device=device)
			sample_rng.manual_seed(42 + ddp_rank)
			while xgen.size(1) < config['sampling']['sample_len']:
				# forward the model to get the logits
				with torch.no_grad():
					with ctx:
						logits, loss = model(xgen)
					logits = logits[:, -1, :] 
					probs = F.softmax(logits, dim=-1)
					topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
					ix = torch.multinomial(topk_probs, 1, generator=sample_rng) 
					xcol = torch.gather(topk_indices, -1, ix) 
					xgen = torch.cat((xgen, xcol), dim=1)
			for i in range(config['sampling']['samples_per_rank']):
				tokens = xgen[i, :config['sampling']['sample_len']].tolist()
				decoded = tokenizer.decode(tokens)
				logger.info(f"rank {ddp_rank} sample {i}: {decoded}")

		if config['general']['val'] and (step % config['validation']['val_every_n_steps'] == 0):
			model.eval()
			val_loss = 0.0
			val_iter = iter(val_loader)
			for val_step in range(config['validation']['val_micro_steps']):
				x, y = next(val_iter)
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
				with ctx:
					logits, loss = model(x, y)
				loss = loss / config['validation']['val_micro_steps']
				val_loss += loss.detach()
			if ddp:
				dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
			if master_process:
				logger.info(f'{step} validation loss: {val_loss.item():.4f} ')

		if  config['general']['eval'] and config['evaluation']['hellaswag'] and (step % config['evaluation']['hell_every_n_steps'] == 0) :
			model.eval()
			hellaswag = HellaSwag(config, ddp_rank, ddp_world_size, device)
			for batch in hellaswag.batch_iterator():
				tokens, mask, labels = hellaswag.prepare_batch(batch)
				tokens, mask = tokens.to(device), mask.to(device)
				with ctx:
					logits, loss = model(tokens)
				hellaswag.count_correct(logits, tokens, mask, labels)
			num_total, correct_norm = hellaswag.get_counts()
			if ddp:
				dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
				dist.all_reduce(correct_norm, op=dist.ReduceOp.SUM)
				num_total, correct_norm = num_total.item(), correct_norm.item()
			if master_process:
				logger.info(f'{step} HellaSwag Result = {correct_norm}/{num_total}={(correct_norm/num_total):.4f}')

		if config['general']['train'] is False:
			break
		loss_accumulation = 0.0
		t0 = time()
		model.train()
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
		dist.all_reduce((loss_accumulation), op=dist.ReduceOp.AVG)
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
	scaler.step(optimizer)
	scheduler.step()
	scaler.update()
	if device == 'cuda':
		torch.cuda.synchronize()

	if master_process:
		tokens_per_sec = (mini_batch * block_size * grad_accum_steps * ddp_world_size) / (time() - t0)
		logger.info(
			f'{step} loss: {loss_accumulation.item()} | iter time: {(time() - t0) * 1000:.2f} ms | lr: {scheduler.get_lr()[0]:.4f} | {tokens_per_sec:.2f} tokens/sec'
		)
		# saving
		if (config['saving']['save_checkpoints'] and step % config['saving']['save_every_n_batches'] == 0 and (step > 0)) or (config['saving']['save_end_model'] and step == config['optimizer']['max_steps']-1):
			raw_model = model.module if ddp else model
			state = {'step': step, 'epoch': epoch, 'config': config, 'model': raw_model.state_dict(), 'loss': loss_accumulation.item()}
			if config['saving']['save_with_resume_option']:
				state['optimizer'] = optimizer.state_dict()
				state['scheduler'] = scheduler.state_dict()
			torch.save(state, f'checkpoint_{step/grad_accum_steps/step_per_epoch:.4f}.pth')
			logger.info(
				f'Checkpoint saved. On {step/grad_accum_steps/step_per_epoch:.4f} epoch. Resumeable save: { str(config["saving"]["save_with_resume_option"])}'
			)


if ddp:
	dist.destroy_process_group()
	if master_process:
		logger.debug('DDP destroyed sucessfuly')
