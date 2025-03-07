from typing import Generator
import requests
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

PATH = 'benchmarks/hellaswag'


class HellaSwag():
	def __init__(self, config: dict[str, dict[str, int]], ddp_rank, ddp_world_size, device):
		self.config = config
		self.device = device
		self.ddp_rank = ddp_rank
		self.ddp_world_size = ddp_world_size
		assert os.path.isfile(f"{PATH}/{config['data']['hellaswag_file']}"), f"In benchmarks/hellaswag directory there is no file {config['data']['hellaswag_file']}"
		self.num_total, self.correct, self.correct_norm = 0,0,0


	def batch_iterator(self) -> Generator[dict[str, list[str]], None, None]:
		with open(f"{PATH}/{self.config['data']['hellaswag_file']}", "r", encoding="utf-8") as file:
			data = [json.loads(line) for line in file]
		n_examples_in_batch = self.config['training']['mini_batch']//4
		perfect_size = len(data) - (len(data) % (self.ddp_world_size * n_examples_in_batch))
		data = data[:perfect_size]
		subset = data[self.ddp_rank::self.ddp_world_size]

		for i in range(0, len(subset), n_examples_in_batch):
			yield subset[i:i + n_examples_in_batch]


	def prepare_batch(self, examples: list[dict[str, list[str]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		ctx = [example['ctx'] for example in examples]
		endings_list = [example['endings'] for example in examples]
		label = [int(example['label']) for example in examples]
		
		endings = [' '+end for endings in endings_list for end in endings]

		tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, clean_up_tokenization_spaces = False)
		ctx = tokenizer.batch_encode_plus(ctx)['input_ids']
		endings = tokenizer.batch_encode_plus(endings)['input_ids']

		ctx = [ct for ct in ctx for _ in range(4)]
		token_rows = [opening + end for opening, end in zip(ctx, endings)]
		mask_rows = [[0] * len(opening) + [1] * len(end) for opening, end in zip(ctx, endings)]

		token_tensors = [torch.tensor(row, dtype=torch.long) for row in token_rows]
		mask_tensors  = [torch.tensor(row, dtype=torch.long) for row in mask_rows]
		tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)
		mask   = pad_sequence(mask_tensors, batch_first=True, padding_value=0)


		if tokens.shape[1] < self.config['model']['block_size']:
			pad_size = self.config['model']['block_size'] - tokens.shape[1]
			tokens = torch.nn.functional.pad(tokens, (0, pad_size), mode='constant', value=0)
			mask   = torch.nn.functional.pad(mask, (0, pad_size), mode='constant', value=0)
		else:
			tokens = tokens[:, :self.config['model']['block_size']]
			mask   = mask[:, :self.config['model']['block_size']]

		return tokens, mask, label


	def calculate_sum_loss(self, logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], list[float]]:
		# evaluate the autoregressive loss at all positions
		shift_logits = (logits[:, :-1, :]).contiguous()
		shift_tokens = (tokens[:, 1:]).contiguous()
		flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
		flat_shift_tokens = shift_tokens.view(-1)
		shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
		shift_losses = shift_losses.view(tokens.size(0), -1)
		# now get the average loss just for the completion region (where mask == 1), in each row
		shift_mask = (mask[:, 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
		masked_shift_losses = shift_losses * shift_mask
		# sum and divide by the number of 1s in the mask
		sum_loss = masked_shift_losses.sum(dim=1)
		avg_loss = sum_loss / shift_mask.sum(dim=1)
		# now we have a loss for each of the 4 completions
		# the one with the lowest loss should be the most likely

		return sum_loss, avg_loss
	
	def count_correct(self, logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor, labels: list[int]):
		sum_loss, avg_loss = self.calculate_sum_loss(logits, tokens, mask)
		pred_norm = avg_loss.view(-1,4).argmin(dim=1)
		self.num_total += len(pred_norm)
		self.correct_norm += (pred_norm == torch.tensor(labels, device = self.device)).sum().item()

	def get_counts(self):
		if self.ddp_world_size > 1:
			self.num_total = torch.tensor(self.num_total, dtype=torch.long, device=self.device)
			self.correct_norm =torch.tensor(self.correct_norm, dtype=torch.long, device=self.device)
		return self.num_total, self.correct, self.correct_norm
