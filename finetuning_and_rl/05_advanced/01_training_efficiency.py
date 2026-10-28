"""Accumulation, mixed precision, clipping, scheduling, checkpointing.
Effective batch = microbatch * accumulation_steps * number_of_data_parallel_workers.
Accumulation saves activation memory, not model/optimizer memory. It matches a
large batch only with compatible reduction, randomness and batch-dependent layers.
Exercise: compare this FP32 update with a single full-batch update.
"""
from contextlib import nullcontext
from pathlib import Path
import math
import torch
from torch import nn


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    # FP16 needs loss scaling to reduce underflow. BF16 often does not.
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    x, y = torch.randn(36, 8, device=device), torch.randn(36, 1, device=device)
    microbatch, accumulation = 4, 4
    microbatches = [(x[i:i+microbatch], y[i:i+microbatch]) for i in range(0, len(x), microbatch)]
    total_updates = math.ceil(len(microbatches)/accumulation)
    updates = 0
    for start in range(0, len(microbatches), accumulation):
        group = microbatches[start:start+accumulation]
        samples = sum(len(a) for a, _ in group)
        optimizer.zero_grad(set_to_none=True)
        # Warmup then cosine decay, stepped per optimizer update, not microbatch.
        warmup = 1
        multiplier = (updates+1)/warmup if updates < warmup else 0.5*(1+math.cos(
            math.pi*(updates-warmup)/max(1, total_updates-warmup)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.003*multiplier
        for inputs, labels in group:
            context = torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()
            with context:
                loss = (model(inputs)-labels).square().mean()
                # Weight by actual sample count, including the final short group.
                # Token losses instead need weighting by supervised token count.
                loss = loss*len(inputs)/samples
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Clip real gradients, not scaled gradients.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        updates += 1
    output = Path(__file__).resolve().parents[1]/'outputs/training_checkpoint.pt'
    output.parent.mkdir(exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'updates': updates,
                'torch_rng': torch.get_rng_state()}, output)
    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    print('saved and reloaded checkpoint:', output)
    # Exact full-run resume also needs sampler position, Python/NumPy/CUDA RNG,
    # scheduler state and data preprocessing/config versions.

if __name__ == '__main__':
    main()
