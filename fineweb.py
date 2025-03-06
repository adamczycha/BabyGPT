import os
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset, DatasetDict
import psutil
import gc
from transformers import AutoTokenizer

# number of workers in .map() call
# good number to use is ~order number o cpu cores // 2
num_proc = psutil.cpu_count(logical=False)
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

if __name__ == '__main__':
	
	dataset = load_dataset('HuggingFaceFW/fineweb-2', name='pol_Latn', split='train', num_proc=num_proc, data_files = ['data/pol_Latn/train/000_00000.parquet', 'data/pol_Latn/train/000_00001.parquet', 'data/pol_Latn/train/000_00002.parquet', 'data/pol_Latn/train/000_00003.parquet', 'data/pol_Latn/train/000_00004.parquet', 'data/pol_Latn/train/000_00005.parquet'])
	dataset = dataset.select_columns(['text'])
	shutil.rmtree('/root/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-2')

	split_dataset = dataset.train_test_split(test_size=0.0005, seed=0, shuffle=False, writer_batch_size=10000)
	split_dataset['val'] = split_dataset.pop('test')
	del dataset
	gc.collect()
	
	
	def tokenize(example: dict[str, list[str]]) -> dict[str, object]:
		batch = tokenizer(example["text"], truncation=False, add_special_tokens=False)
		tokens = [ids + [tokenizer.eos_token_id] for ids in batch["input_ids"]]
		lengths = [len(i) for i in tokens]
		out = {'ids': tokens, 'len': lengths}
		return out

	tokenized = split_dataset.map(
		tokenize,
		remove_columns=['text'],
		batched=True,
		desc='tokenizing data',
		num_proc=num_proc,
	)
	del split_dataset
	gc.collect()

	for split, dset in tokenized.items():
		arr_len = np.sum(dset['len'], dtype=np.uint64)
		filename = os.path.join('/workspace', f'{split}.bin')
		dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
		arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
		total_batches = 1024

		idx = 0
		for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
			# Batch together samples for faster write
			batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
			arr_batch = np.concatenate(batch['ids'])
			# Write into mmap
			arr[idx : idx + len(arr_batch)] = arr_batch
			idx += len(arr_batch)
		arr.flush()
