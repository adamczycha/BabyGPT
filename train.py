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
from src.hellaswag import iterate_examples, calculate_sum_loss, prepare_example
import yaml
import tiktoken
import torch.functional as F


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
	assert os.path.exists(model_dir), 'You want to resume training, but file does not exist!'

	checkpoint = torch.load(model_dir)
	config['model'] = checkpoint['config']['model']
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
	logger.info('GPT from pretrained loaded.')

elif config['training']['init_from'] == 'scratch':
	model = GPT(config)
	model.to(device)
	optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=0, device=device)
	scheduler = CosineScheduler(optimizer, config)  # does not depend on optimizer learnig rate


train_config = config['training']
model_config = config['model']
model_eval = config['evaluation']
batch_size = train_config['batch_size']
mini_batch = train_config['mini_batch']
block_size = model_config['block_size']


try:
	assert batch_size % (mini_batch * block_size * ddp_world_size) == 0
except AssertionError:
	if master_process:
		logger.critical(f'BATCH_SIZE is not divisible by mini_batch * block_size * world_size. Batch_size => ({mini_batch * block_size * ddp_world_size})')
grad_accum_steps = batch_size // (mini_batch * block_size * ddp_world_size)
if master_process:
	logger.info(f'total desiered batch size {batch_size}')
	logger.info(f'=> calculated in gradient accumation steps: {grad_accum_steps}')


if torch.cuda.is_available() and train_config['compile']:
	model.compile()

train_dataset = TokenDataset(config=config, split='train')
sampler = ChankSampler(config=config, dataset=train_dataset, shuffle=True, seed=0)
train_loader = DataLoader(
	dataset=train_dataset, batch_size=(mini_batch * block_size), sampler=sampler, collate_fn=custom_collate_fn, pin_memory=True
)

val_dataset = TokenDataset(config=config, split='val')
val_sampler = ChankSampler(config, dataset=val_dataset, split='validation')
val_loader = DataLoader(
	dataset=val_dataset, batch_size=(mini_batch * block_size), sampler=val_sampler, collate_fn=custom_collate_fn, pin_memory=True
)

if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])

step_per_epoch = len(train_dataset) // batch_size
if master_process:
	logger.info(f' {step_per_epoch} batches in epoch')

epoch = -1 if 'checkpoint' not in locals() else checkpoint['epoch']
last_step = 0 if 'checkpoint' not in locals() else checkpoint['step'] + 1
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

for step in range(last_step, int(config['optimizer']['max_steps'])):
	if step % (step_per_epoch - 1) == 0 or resume_run:
		epoch += 1
		sampler.set_epoch(epoch)
		train_iter = iter(train_loader)
		resume_run = False

	with (model.no_sync() if ddp else nullcontext()):
		# once in a while generate from the model (except step 0, which is noise)
		if (((step > 0 and step % model_eval['sample_every_n'] == 0) or last_step)) and train_config['sample']:
			model.eval()
			num_return_sequences = 4
			max_length = 32
			enc = tiktoken.get_encoding('gpt2')
			tokens = enc.encode("Reynevan w raz z Szarlej")
			tokens = torch.tensor(tokens, dtype=torch.long)
			tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
			xgen = tokens.to(device)
			sample_rng = torch.Generator(device=device)
			sample_rng.manual_seed(42 + ddp_rank)
			while xgen.size(1) < max_length:
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
			for i in range(num_return_sequences):
				tokens = xgen[i, :max_length].tolist()
				decoded = enc.decode(tokens)
				print(f"rank {ddp_rank} sample {i}: {decoded}")

		if step % model_eval['validation_every_n_steps'] == 0:
			model.eval()
			val_loss = 0.0
			val_iter = iter(val_loader)
			for val_step in range(model_eval['validation_micro_steps']):
				x, y = next(val_iter)
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
				with ctx:
					logits, loss = model(x, y)
				loss = loss / model_eval['validation_micro_steps']
				val_loss += loss.detach()
			if ddp:
				dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
			logger.info(f'validation loss: {val_loss.item():.4f} ')

		if step % model_eval['hellaswag_every_n_steps'] == 0 and train_config['eval']:
			model.eval()
			num_total = 0
			num_correct_norm = 0
			examples = iterate_examples()
			for i, example in enumerate(examples):
				if i % ddp_world_size == ddp_rank:
					tokens, mask, label = prepare_example(example)
					tokens, mask = tokens.to(device), mask.to(device)
					with ctx:
						logits, loss = model(tokens)
					sum_loss, avg_loss = calculate_sum_loss(logits, tokens, mask)
					pred = sum_loss.argmin().item()
					pred_norm = avg_loss.argmin().item()
					num_total += 1
					num_correct_norm += int(pred_norm == label)
			if ddp:
				num_total = torch.tensor(num_total, dtype=torch.long, device=device)
				num_correct_norm = torch.tensor(num_correct_norm, device=device)
				dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
				dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
				num_total.item()
				num_correct_norm.item()
			if master_process:
				logger.info(f'HellaSwag step {step}. Result = {num_correct_norm}/{num_total}={(num_correct_norm/num_total):.4f}')

		if config['training']['eval_only'] is True:
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
		dist.all_reduce(loss_accumulation, op=dist.ReduceOp.AVG)
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
		saving_config = config['saving']
		# saving
		if saving_config['save_checkpoints'] and step % saving_config['save_every_n_batches'] == 0:
			raw_model = model.module if ddp else model
			state = {'step': step, 'epoch': epoch, 'config': config, 'model': raw_model.state_dict(), 'loss': loss_accumulation.item()}
			if saving_config['save_with_resume_option']:
				state['optimizer'] = optimizer.state_dict()
				state['scheduler'] = scheduler.state_dict()
			torch.save(state, f'checkpoint_{step/grad_accum_steps/step_per_epoch:.4f}.pth')
			logger.info(
				f'Checkpoint saved. On {step/grad_accum_steps/step_per_epoch:.4f} epoch. Resumeable save: { str(saving_config["save_with_resume_option"])}'
			)


if ddp:
	dist.destroy_process_group()
	logger.debug('DDP destroyed sucessfuly')
