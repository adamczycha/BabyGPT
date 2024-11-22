from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from typing import Optional, Tuple


@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_head: int = 12
	n_layer: int = 12
	n_embd: int = 768

	def __init__(self, config: dict[str, int]) -> None:
		super().__init__()
		if isinstance(config, dict):
			self.block_size = config['block_size']
			self.vocab_size = config['vocab_size']
			self.n_head = config['n_head']
			self.n_layer = config['n_layer']
			self.n_embd = config['n_embd']


class CasualSelfAttention(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()

		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
		self.c_proj = nn.Linear(config.n_embd, config.n_embd)
		self.c_proj.NANOGPT_SCALE_INIT = 1
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.register_buffer(
			'bias',
			torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, T, C = x.size()

		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

		#         att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
		#         att = att.masked_fill(self.bias[:,:,:T,:T] == 0, (-np.inf))
		#         att = F.softmax(att, dim = -1)
		#         y = att @ v # (B, nh, T, hs)
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.c_proj(y)
		return y


class MLP(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
		self.gelu = nn.GELU(approximate='tanh')
		self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x


class Block(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.attn = CasualSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPT(nn.Module):
	def __init__(self, config: dict[str, dict[str, int]] | GPTConfig) -> None:
		super().__init__()
		self.config = config if isinstance(config, GPTConfig) else GPTConfig(config['model'])

		self.transformer = nn.ModuleDict(
			dict(
				wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
				wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
				h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
				ln_f=nn.LayerNorm(self.config.n_embd),
			)
		)
		self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

		self.transformer.wte.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module: nn.Module) -> None:
		if isinstance(module, nn.Linear):
			std = 0.02
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx: torch.Tensor, targets: torch.Tensor | None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		B, T = idx.size()
		assert T <= self.config.block_size, f'Model cannot operate {T} as a block size maximum is {self.config.block_size}'
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
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
	def from_pretrained(cls, model_type: str) -> 'GPT':
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		from transformers import GPT2LMHeadModel

		print(f'loading weights from pretrained gpt: {model_type}')
		config_args = {
			'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
			'gp2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
			'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
			'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
		}[model_type]
		config_args['vocab_size'] = 50257
		config_args['block_size'] = 1024
		config = GPTConfig(config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

		# init a huggingface/transformers model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
		transposed = [
			'attn.c_attn.weight',
			'attn.c_proj.weight',
			'mlp.c_fc.weight',
			'mlp.c_proj.weight',
		]
		# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
		# this means that we have to transpose these weights when we import them
		assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
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

	def configure_optimizer(self, weight_decay: float, learning_rate: float, device: str) -> torch.optim.Optimizer:
		param_dict = {name: param for name, param in self.named_parameters()}
		param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

		decay_params = [param for param in param_dict.values() if param.dim() >= 2]
		nodecay_params = [param for param in param_dict.values() if param.dim() < 2]
		optim_groups = [
			{'params': decay_params, 'weight_decay': weight_decay},
			{'params': nodecay_params, 'weight_decay': 0.0},
		]
		num_decayed_params = sum(param.numel() for param in decay_params)
		num_nodecayed_params = sum(param.numel() for param in nodecay_params)
		print(f'num decayed parameter tensors {len(decay_params)}, with {num_decayed_params} weights')
		print(f'num no decayed paramter tensors {len(nodecay_params)}, with {num_nodecayed_params} weights')
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and 'cuda' in device
		print(f'using fused AdamW: {use_fused}')
		optimizer = torch.optim.AdamW(
			optim_groups,
			betas=(0.9, 0.95),
			eps=10e-8,
			lr=learning_rate,
			fused=use_fused,
		)
		return optimizer
