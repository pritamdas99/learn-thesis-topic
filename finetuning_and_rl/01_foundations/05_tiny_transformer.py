"""Train and sample a tiny causal transformer entirely offline.
This is pretraining from random initialization, NOT fine-tuning a pretrained LLM.
A single repeated sentence demonstrates mechanics, not language competence.
Exercise: use a richer corpus, split it BEFORE making overlapping windows,
and report validation cross-entropy instead of memorization loss.
"""
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size, width=32, context=24):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(width, 4, 64, dropout=0.0, batch_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids):
        length = ids.shape[1]
        x = self.tokens(ids) + self.positions(torch.arange(length, device=ids.device))
        mask = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
        return self.head(self.blocks(x, mask=mask))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    corpus = 'a cat sits on a mat. ' * 40
    chars = sorted(set(corpus))
    encoded = torch.tensor([chars.index(c) for c in corpus])
    model = TinyLM(len(chars))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    for step in range(120):
        starts = torch.randint(len(encoded)-model.context-1, (16,))
        batch = torch.stack([encoded[s:s+model.context+1] for s in starts])
        logits = model(batch[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(chars)), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 40 == 0:
            print('step:', step, 'training loss:', round(loss.item(), 3))
    model.eval()
    ids = torch.tensor([[chars.index('a')]])
    with torch.no_grad():
        for _ in range(60):
            logits = model(ids[:, -model.context:])[:, -1]
            # Temperature <1 sharpens probabilities; >1 increases randomness.
            next_id = torch.multinomial((logits / 0.8).softmax(-1), 1)
            ids = torch.cat([ids, next_id], dim=1)
    print('sample:', ''.join(chars[i] for i in ids[0].tolist()))

if __name__ == '__main__':
    main()
