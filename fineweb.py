import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import psutil
import gc

# number of workers in .map() call
# good number to use is ~order number o cpu cores // 2
num_proc = psutil.cpu_count(logical=False)
enc = tiktoken.get_encoding('gpt2')

if __name__ == '__main__':
	dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train')
	split_dataset = dataset.train_test_split(test_size=0.0005, seed=0, shuffle=True, writer_batch_size=10000)
	split_dataset['val'] = split_dataset.pop('test')
	del dataset
	gc.collect()

	def tokenize(example: dict[str,str ]) -> dict[str, int]:
		ids = enc.encode_ordinary(example['text'])
		ids.append(enc.eot_token)
		out = {'ids': ids, 'len': len(ids)}
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
		filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
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
