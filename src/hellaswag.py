from typing import Generator
import requests
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
from torch.nn import functional as F

PATH = 'benchmarks/hellaswag'




def iterate_examples(config: dict[str, dict[str, int]]) -> Generator[dict[str, list[str]], None, None]:
	assert os.path.isfile(f'{PATH}/{config['data']['hellaswag_file']}'), f'In benchmarks/hellaswag directory there is no file {config['data']['hellaswag_file']}'
	with open(f'{PATH}/{config['data']['hellaswag_file']}', 'rb') as file:
		for line in file:
			example = json.loads(line)
			yield example


def prepare_example(example: dict[str, list[str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	ctx = example['ctx']
	endings = example['endings']
	label = example['label']

	tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
	ctx_tokens = tokenizer.encode(ctx)
	tok_rows = []
	mask_rows = []
	for end in endings:
		end_tokens = tokenizer.encode(' ' + end)
		tok_rows.append(ctx_tokens + end_tokens)
		mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

	tokens = torch.zeros([4, 1024], dtype=torch.long)
	mask = torch.zeros([4, 1024], dtype=torch.long)
	for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
		tokens[i, : len(tok_row)] = torch.tensor([tok_row])
		mask[i, : len(mask_row)] = torch.tensor([mask_row])

	return tokens, mask, label


def calculate_sum_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], list[float]]:
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


@torch.no_grad()
def evaluate(model_type: str, device: str) -> None:
	torch.set_float32_matmul_precision('high')  # use tf32
	model = AutoModelForCausalLM.from_pretrained(model_type)
	model.to(device)

	num_correct_norm = 0
	num_correct = 0
	num_total = 0
	for example in iterate_examples():
		tokens, mask, label = prepare_example(example)
		tokens = tokens.to(device)
		mask = mask.to(device)
		logits = model(tokens).logits
		sum_loss, avg_loss = calculate_sum_loss(logits, tokens, mask)
		pred = sum_loss.argmin().item()
		pred_norm = avg_loss.argmin().item()
		# accumulate stats
		num_total += 1
		num_correct += int(pred == label)
		num_correct_norm += int(pred_norm == label)
		print(f'{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}')

		# debug: pretty print a few examples, and the losses in each case
		if num_total < 10:
			print(f"Context:\n {example['ctx']}")
			print('Endings:')
			for i, end in enumerate(example['endings']):
				print(f'{i} (loss: {avg_loss[i].item():.4f}) {end}')
			print(f'predicted: {pred_norm}, actual: {label}')
			print('---')


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_type', type=str, default='gpt2', help='the model type to use')
	parser.add_argument('-d', '--device', type=str, default='cuda', help='the device to use')
	args = parser.parse_args()
	evaluate(args.model_type, args.device)
