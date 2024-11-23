from torch.utils.data._utils.collate import default_collate
from typing import Tuple, List
import torch
import yaml

with open('train_config.yaml', 'r') as file:
	config = yaml.safe_load(file)
mini_batch = config['training']['mini_batch']
block_size = config['model']['block_size']


def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
	collated_batch = default_collate(batch)
	collated_batch = [collated_batch[0].view(mini_batch, block_size), collated_batch[1].view(mini_batch, block_size)]
	return collated_batch  # type: ignore
