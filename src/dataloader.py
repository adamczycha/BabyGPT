from torch.utils.data._utils.collate import default_collate
from configparser import ConfigParser

config = ConfigParser()
config.read('train.cfg')
mini_batch = config['training']['mini_batch']
block_size = config['model']['block_size']


def custom_collate_fn(batch):
    collated_batch = default_collate(batch)
    collated_batch = [collated_batch[0].view(mini_batch, block_size),collated_batch[1].view(mini_batch, block_size)]
    return collated_batch