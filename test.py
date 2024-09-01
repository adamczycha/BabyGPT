import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Set up the process group
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)
    torch.manual_seed(42)

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to the corresponding device
    model = torch.nn.Linear(10, 10).to(rank)  # Assuming model is simple for demonstration

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Dummy optimizer
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    # Dummy input and target
    inputs = torch.randn(20, 10).to(rank)
    targets = torch.randn(20, 10).to(rank)

    # Training loop
    for _ in range(10):  # Small number of iterations for demonstration
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

    # Clean up
    dist.destroy_process_group()

def spawn_processes():
    world_size = 4  # Number of processes, adjust according to your CPU cores

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Pick an available port
    os.environ['WORLD_SIZE'] = str(world_size)

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    spawn_processes()
