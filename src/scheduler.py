import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from configparser import ConfigParser

class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, config: ConfigParser):
        optim_conf = config['optimizer']
        self.warmup_steps = int(optim_conf['warmup_steps'])
        self.max_steps = int(optim_conf['max_steps'])
        self.max_lr = float(optim_conf['max_lr'])
        self.min_lr = self.max_lr = float(optim_conf['min_lr'])
       
        self.last_epoch = 0
        super(_LRScheduler, self).__init__(optimizer)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_steps:
            # Warm-up phase
            return [self.max_lr * (it + 1) / self.warmup_steps for _ in self.base_lrs]
        if it >= self.max_steps:
            # Beyond max_steps, return min_lr
            return [self.min_lr for _ in self.base_lrs]

        # Cosine decay phase
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [
            self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.base_lrs
        ]
