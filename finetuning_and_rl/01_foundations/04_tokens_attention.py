"""Token IDs, embeddings, causal attention, and next-token labels.
Real tokenizers usually use subword units; characters make the mechanics clear.
Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V. A causal mask blocks the future.
Exercise: remove the mask and explain why the resulting training task leaks.
"""
import math
import torch
from torch import nn


def causal_attention(x):
    # Here Q=K=V=x for clarity; a transformer learns separate projections.
    length, dimension = x.shape
    scores = x @ x.T / math.sqrt(dimension)
    future = torch.ones(length, length, dtype=torch.bool).triu(diagonal=1)
    probabilities = scores.masked_fill(future, float('-inf')).softmax(-1)
    return probabilities @ x, probabilities


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    text = 'cat sat'
    vocabulary = sorted(set(text))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    ids = torch.tensor([token_to_id[c] for c in text])
    embedding = nn.Embedding(len(vocabulary), 8)
    inputs, labels = ids[:-1], ids[1:]  # Each position predicts the next character.
    output, attention = causal_attention(embedding(inputs))
    print('vocabulary:', token_to_id)
    print('input IDs:', inputs.tolist(), 'next-token IDs:', labels.tolist())
    print('attention weights:', attention.detach().round(decimals=2))
    assert torch.all(attention.triu(1) == 0)
    assert torch.allclose(attention.sum(-1), torch.ones(len(inputs)))
    print('output shape:', output.shape)
    # Full transformers also use positional information, multiple heads,
    # residual additions, normalization and feed-forward blocks: see 05_tiny_transformer.py.

if __name__ == '__main__':
    main()
