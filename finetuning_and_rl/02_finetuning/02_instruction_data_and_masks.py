"""Instruction examples, response-only loss, padding, and leakage.
SFT = supervised fine-tuning: maximize likelihood of desired responses.
Prompt tokens provide context; only response tokens contribute to this loss.
-100 is PyTorch cross-entropy's default ignored target. It is NOT a token ID.
Exercise: add a second example of different length and inspect every label.
"""
import torch
from torch import nn


def make_batch(examples):
    # A byte tokenizer is only a teaching aid. IDs 256 and 257 are EOS and PAD.
    # Real chat models need their OWN tokenizer and chat template.
    eos, pad = 256, 257
    sequences, labels = [], []
    for prompt, answer in examples:
        prefix = list(('User: ' + prompt + '\nAssistant: ').encode('utf-8'))
        completion = list(answer.encode('utf-8')) + [eos]
        sequences.append(prefix + completion)
        labels.append([-100]*len(prefix) + completion)
    length = max(map(len, sequences))
    ids = torch.tensor([s + [pad]*(length-len(s)) for s in sequences])
    targets = torch.tensor([s + [-100]*(length-len(s)) for s in labels])
    return ids, targets, ids.ne(pad)


def response_loss(logits, labels):
    # logits at position t predict token t+1. Shift EXACTLY ONCE.
    # Hugging Face causal LM forward(labels=...) generally shifts internally.
    return nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    ids, labels, attention_mask = make_batch([('2+2?', '4'), ('Say hello', 'Hello!')])
    print('batch shape:', ids.shape)
    print('supervised tokens per example:', labels.ne(-100).sum(-1).tolist())
    print('padding tokens:', (~attention_mask).sum().item())
    logits = torch.randn(*ids.shape, 258, requires_grad=True)
    loss = response_loss(logits, labels)
    loss.backward()
    assert torch.all(labels[~attention_mask] == -100)
    assert logits.grad.isfinite().all()
    print('response-only loss:', loss.item())
    # Split by document/user/source BEFORE chunking. Deduplicate across splits.
    # An attention mask controls visibility; a label mask controls supervision.
    # If truncation removes every response token, drop that training example.

if __name__ == '__main__':
    main()
