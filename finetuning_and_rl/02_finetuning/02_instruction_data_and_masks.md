# Instruction tuning, response-only loss, and token masks

Companion code: [02_instruction_data_and_masks.py](02_instruction_data_and_masks.py). Read this explanation before running the code.

## What supervised fine-tuning teaches

SFT trains a model to reproduce desired responses to prompts. Given prompt $x$
and response tokens $y_1,\ldots,y_M$, the conditional likelihood is:

$$
p_\theta(y\mid x)=\prod_{t=1}^M p_\theta(y_t\mid x,y_{<t}).
$$

Minimizing its negative logarithm teaches the response while using the prompt
as context. This is supervised learning: desired responses are supplied, unlike
RL where the agent learns from consequences or rewards.

Read [next-token prediction](../01_foundations/05_tiny_transformer.md) first.
This file constructs a batch and checks its loss; it does not train an LLM.

## Tokens, formatting, and boundaries

The teaching tokenizer represents UTF-8 bytes as IDs 0–255, with EOS=256 and
PAD=257. Non-ASCII characters may use several bytes. A real model must use its
own tokenizer and expected prompt/chat format.

For each example, the script concatenates a prefix such as
`User: 2+2?\nAssistant: ` with the answer `4` and an EOS token.
The target labels for the prefix are -100, while labels for the answer and EOS
are their actual token IDs. -100 is a loss sentinel, not a token passed through
the embedding table.

Why supervise EOS? The model needs to learn when to stop. Why not supervise the
prompt? In response-only SFT, the goal is conditional response behavior; prompt
reconstruction would spend part of the objective on a different task.

## Two masks do different jobs

An **attention mask** says which input positions are real rather than padding.
A **label mask** says which next-token predictions contribute to the objective.
A prompt can be visible to attention while excluded from direct loss.

The loss with label mask $m_{bt}$ is:

$$
L=-\frac{\sum_{b,t}m_{bt}\log p_\theta(u_{bt}\mid u_{b,<t})}
{\sum_{b,t}m_{bt}}.
$$

The denominator counts supervised tokens, not all positions or examples.
Consequently, examples with longer responses contribute more token terms.
That is a weighting choice, not an unavoidable property of SFT.

Ignoring prompt labels does **not** stop all gradients through prompt
representations: response predictions depend on them through attention.
It removes direct prompt-token prediction loss.

## Why we shift labels exactly once

Suppose the full IDs are `[10,20,30,256]`, where `[10,20]` is the prompt and
`[30,256]` is the response plus EOS. Labels are `[-100,-100,30,256]`.

| Logit position | Predicts full-sequence position | Supervised target |
| --- | --- | --- |
| 0 | 1 | ignored |
| 1 | 2 | 30, the first response token |
| 2 | 3 | 256, EOS |

The code pairs `logits[:, :-1]` with `labels[:, 1:]`. Notice that the first
answer token is predicted from the final prompt position. Excluding all logits
at prompt positions would incorrectly remove this important prediction.

Hugging Face causal models usually perform this shift internally when passed
`labels`. The local helper performs it manually because it receives raw logits.
Doing both would make every target off by one.

## Map to code and inspect the outcome

`make_batch` creates equal-length ID and label rows. It pads inputs with PAD
and labels with -100. `ids.ne(pad)` creates the attention mask. `response_loss`
flattens batch and sequence dimensions, leaving vocabulary as the class axis,
and calls cross-entropy with ignored targets excluded.

The random logits in `main` are a diagnostic tensor with gradients enabled.
There is no attention model in this file, so the returned attention mask is
inspected, not passed into a neural network. The gradient check confirms the
loss is differentiable; it does not show a model has learned anything.

Run `python 02_finetuning/02_instruction_data_and_masks.py`. The two examples
have 2 and 7 supervised tokens: answer bytes plus EOS. Verify this manually.
Try changing logits whose shifted labels are -100: the loss should not change.
If truncation leaves zero response tokens, drop or fix that example; an empty
supervision denominator can produce an undefined loss.

Finally, split by source before chunking and deduplicate across splits. Perfect
label masking cannot repair train/test leakage in the data itself.
