import numpy as np
import torch
from torch.utils.data import Dataset
from configparser import ConfigParser


class TokenDataset(Dataset):
	def __init__(self, config: ConfigParser, split: str) -> None:
		# dataset path: {dataset}/test/test.bin dtype = int16
		self.config = config
		self.split = split
		self.dataset = config['data']['dataset']
		self.tokens = np.memmap(f'{self.dataset}/{self.split}/{self.split}.bin', dtype=np.uint16, mode='r')

	def __len__(self) -> int:
		return len(self.tokens)

	def __getitem__(self, idx: slice | int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		if isinstance(idx, slice):
			return torch.tensor(self.tokens[idx.start : idx.stop], dtype=torch.long)
		else:
			try:
				return torch.tensor(self.tokens[idx], dtype=torch.long), torch.tensor(self.tokens[idx + 1], dtype=torch.long)
			except IndexError:
				return torch.tensor(self.tokens[idx], dtype=torch.long), torch.tensor(50265, dtype=torch.long)
