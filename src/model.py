from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from typing import Optional, Tuple


@dataclass
@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_head: int = 12
	n_layer: int = 12
	n_embd: int = 768

	def __init__(self, config: Optional[dict[str, int]] = None) -> None:
		if config:
			self.block_size = config.get('block_size', self.block_size)
			self.vocab_size = config.get('vocab_size', self.vocab_size)
			self.n_head = config.get('n_head', self.n_head)
			self.n_layer = config.get('n_layer', self.n_layer)
			self.n_embd = config.get('n_embd', self.n_embd)


class CasualSelfAttention(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.config = config
		self.n_kv_heads = config.n_head // config.kv_group_factor
		self.n_rep = config.kv_group_factor
		self.head_dim = config.n_embd // config.n_head

		self.wq = nn.Linear(config.n_embd, config.n_embd)
		self.wk = nn.Linear(config.n_embd, self.head_dim * self.n_kv_heads)
		self.wv = nn.Linear(config.n_embd, self.head_dim * self.n_kv_heads)

		self.wout = nn.Linear(config.n_embd, config.n_embd)

		self.cache_k = None
		self.cache_v = None
		self.use_kv_cache = False

	def init_kv_cache(self, max_batch_size, max_seq_len, device):
		self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            device=device,
        )
		self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            device=device,
        )
	
	def enable_kv_cache(self, mode=True):
		self.use_kv_cache = mode

	def kv_repeat(self, x: torch.Tensor, n_rep: int) -> list[torch.Tensor]:
		if n_rep == 1:
			return x
		batch_size, seq_len, kv_heads, head_dim = x.shape
		return (x[:, :, :, None, :]
		  .expand(batch_size, seq_len, kv_heads, n_rep , head_dim)
		  .reshape(batch_size, seq_len, kv_heads * n_rep, head_dim))

	def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
		batchsize, seqlen, dim  = x.shape
		xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

		xq = xq.view(batchsize, seqlen, self.config.n_head, dim // self.config.n_head)
		xk = xk.view(batchsize, seqlen, self.n_kv_heads, dim // self.n_kv_heads)
		xv = xv.view(batchsize, seqlen, self.n_kv_heads, dim // self.n_kv_heads)

		if self.use_kv_cache:
			self.cache_k = self.cache_k.to(xq)
			self.cache_v = self.cache_v.to(xq)

			self.cache_k[:batchsize, start_pos : start_pos + seqlen]
			self.cache_v[:batchsize, start_pos : start_pos + seqlen]

			xk = self.cache_k[:batchsize, :start_pos + seqlen]
			xv = self.cache_v[:batchsize, :start_pos + seqlen]

		xk, xv = self.kv_repeat(xk, self.n_rep), self.kv_repeat(xv, self.n_rep)

		xq = xq.transpose(1,2)
		xk = xk.transpose(1,2)
		xv = xv.transpose(1,2)

		y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

		y = y.transpose(1, 2).contiguous().view(batchsize, seqlen, dim)
		y = self.wout(y)
		return y
	
		#         att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
		#         att = att.masked_fill(self.bias[:,:,:T,:T] == 0, (-np.inf))
		#         att = F.softmax(att, dim = -1)
		#         y = att @ v # (B, nh, T, hs)


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
	def __init__(self, config: dict[str, dict[str, int]] | GPTConfig = GPTConfig()) -> None:
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

	def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
