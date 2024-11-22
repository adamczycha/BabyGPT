import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List


class CosineScheduler(_LRScheduler):
	def __init__(self, optimizer: Optimizer, config: dict[str, dict[str, int]]) -> None:
		optim_conf = config['optimizer']
		self.warmup_steps = optim_conf['warmup_steps']
		self.max_steps = optim_conf['max_steps']
		self.max_lr = float(optim_conf['max_lr'])
		self.min_lr = self.max_lr * optim_conf['min_lr']

		super(_LRScheduler, self).__init__(optimizer)

	def get_lr(self) -> List[float]:
		if self._step_count < self.warmup_steps:
			# Warm-up phase
			return [self.max_lr * (self._step_count + 1) / self.warmup_steps for _ in self.base_lrs]
		if self._step_count >= self.max_steps:
			# Beyond max_steps, return min_lr
			return [self.min_lr for _ in self.base_lrs]

		# Cosine decay phase
		decay_ratio = (self._step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
		coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
		return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.base_lrs]
