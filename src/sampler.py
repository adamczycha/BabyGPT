import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, Sampler
from configparser import ConfigParser
from typing import Iterator
from .dataset import TokenDataset

class ChankSampler(Sampler):
	def __init__(self, config: ConfigParser, dataset: TokenDataset, shuffle: bool = True, seed: int = 0):
		self.dataset = dataset
		self.seed = seed
		self.shuffle = shuffle
		self.config = config
		self.epoch = 0

	def __iter__(self) -> Iterator[int]:
		document_indices = self.locate_document_EOF_to_create_chank()

		# drop last to fit ideally for multiple GPU run
		ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
		mini_batch = int(self.config['training']['mini_batch'])
		block_size = int(self.config['model']['block_size'])

		if self.shuffle:
			random.seed(self.seed + self.epoch)
			random.shuffle(document_indices)

		document_indices = self.drop_last_in_every_document_stream(document_indices, ddp_world_size, mini_batch, block_size)

		ddp_rank = int(os.environ.get('RANK', 0))
		for chank_start, chank_end in document_indices[ddp_rank::ddp_world_size]:
			for idx in range(chank_start, chank_end):
				yield idx

	def set_epoch(self, epoch: int) -> None:
		self.epoch = epoch

	def locate_document_EOF_to_create_chank(
		self, cursor: int = 30000000, step_size: int = 30000000, search_range: int = 20000
	) -> list[tuple[int, int]]:
		document_boundry: list[int] = []
		while cursor < len(self.dataset):
			# Search for EOF token in a fixed range around the cursor\
			EOF_list: list[int] = []
			while EOF_list == []:
				search_start = max(cursor - search_range, 0)
				search_end = min(cursor + search_range, len(self.dataset))
				EOF_list = np.where(self.dataset.tokens[search_start:search_end] == 50256)[0]

				if EOF_list.size == 0:
					search_start += search_range
					search_end += search_range
				else:
					EOF_index = search_start + EOF_list[0]
					document_boundry.append(EOF_index)
					break
			cursor += step_size

		# Add the start and end boundaries
		document_boundry.insert(0, 0)
		document_boundry.append(len(self.dataset))

		# Create document indices
		return [(document_boundry[i], document_boundry[i + 1]) for i in range(len(document_boundry) - 1)]

	def drop_last_in_every_document_stream(
		self, document_indices: list[tuple[int, int]], ddp_world_size: int, mini_batch: int, block_size: int
	) -> list[tuple[int, int]]:
		doc_length = lambda x: x[1] - x[0]
		length_data_per_rank: list[int] = []
		# calculate length for every stream
		for rank in range(ddp_world_size):
			documents_ranges = list(document_indices[rank::ddp_world_size])
			length_data_per_rank.append(sum(map(doc_length, documents_ranges)))
		max_token_length = min(length_data_per_rank) - (min(length_data_per_rank) % (mini_batch * block_size))
		index_remove = []
		# drop or shorten documents to fit perfectly into mini_batch * block_size
		for rank in range(ddp_world_size):
			indices = list(range(rank, len(document_indices), ddp_world_size))
			drop_tokens = 0
			for idx in indices[::-1]:
				drop_tokens += doc_length(document_indices[idx])
				if length_data_per_rank[rank] - drop_tokens <= max_token_length:
					start, end = document_indices[idx]
					lacking_tokens = max_token_length - (length_data_per_rank[rank] - drop_tokens)
					end = start + lacking_tokens
					document_indices[idx] = (start, end)
					break
				else:
					index_remove.append(idx)

		for idx in sorted(index_remove, reverse=True):
			del document_indices[idx]

		return document_indices


