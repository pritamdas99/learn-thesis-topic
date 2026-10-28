"""Minimal CPU DistributedDataParallel training, two processes.
Run from the course directory:
  torchrun --standalone --nproc_per_node=2 05_advanced/07_distributed_training.py
DDP replicates the model, splits data across workers, and averages gradients.
It does NOT shard model weights/optimizer state; FSDP/ZeRO address that memory.
This CPU/gloo lesson avoids GPU assumptions; use NCCL and one GPU per rank for CUDA.
Exercise: print sample indices per rank and verify that training data is partitioned.
"""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def main():
    if 'RANK' not in os.environ:
        raise SystemExit('Launch with torchrun --standalone --nproc_per_node=2 '+__file__)
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    try:
        torch.manual_seed(7)
        x = torch.randn(128, 4)
        dataset = TensorDataset(x, x.sum(-1, keepdim=True))
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)
        model = DistributedDataParallel(nn.Linear(4, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for epoch in range(10):
            sampler.set_epoch(epoch)  # Different deterministic shuffling each epoch.
            for inputs, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = (model(inputs)-targets).square().mean()
                loss.backward()  # DDP synchronizes gradient buckets here.
                optimizer.step()
        if dist.get_rank() == 0:
            print('DDP learned weights:', model.module.weight.detach().tolist())
        # Save only on rank zero. Aggregate evaluation metrics across ranks.
        # Uneven local batch sizes require care: rank means != global sample mean.
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
