"""Distillation and the arithmetic behind simple quantization.
Distillation teaches a student using teacher probabilities (soft targets).
This is symmetric per-tensor INT8 fake quantization, NOT NF4 or full QLoRA.
Exercise: compare per-tensor with per-row scales on weights with outliers.
"""
import torch
from torch import nn


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
    # PyTorch kl_div takes LOG student probabilities and ordinary teacher probs.
    soft = nn.functional.kl_div(
        (student_logits/temperature).log_softmax(-1),
        (teacher_logits.detach()/temperature).softmax(-1), reduction='batchmean')
    # T^2 compensates for the temperature's effect on gradient scale.
    hard = nn.functional.cross_entropy(student_logits, labels)
    return alpha*temperature**2*soft+(1-alpha)*hard


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    teacher = torch.tensor([[4., 2., -1.], [-1., 2., 4.]])
    student = nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.Adam([student], lr=0.05)
    for _ in range(100):
        loss = distillation_loss(student, teacher, torch.tensor([0, 2]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    weights = torch.randn(8, 8)
    scale = weights.abs().max().clamp_min(1e-8)/127
    integers = (weights/scale).round().clamp(-127, 127).to(torch.int8)
    reconstructed = integers.float()*scale
    print('student probabilities:', student.softmax(-1).detach().tolist())
    print('quantization MSE:', (weights-reconstructed).square().mean().item())
    # This stores INT8 weights but dequantizes before arithmetic; accelerated
    # inference needs appropriate quantized kernels, packing, and hardware.

if __name__ == '__main__':
    main()
