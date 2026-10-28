# Fine-tuning a pretrained language model: objective, data, and updates

Companion code: [04_huggingface_sft.py](04_huggingface_sft.py). Read this explanation before running the code.

## What this workflow does

This is the real-model version of [response-only SFT](02_instruction_data_and_masks.md).
It downloads a pretrained causal model, formats local prompt/response pairs,
trains either all weights or LoRA adapters, measures validation loss, and saves
the result. Unlike the tiny transformer lesson, it starts from existing weights.
The bundled eight examples teach the workflow, not a useful domain assistant.

The default is a base completion model, DistilGPT-2. It is not a chat model.
Changing the model ID may require changing tokenization and prompt formatting;
a model's conversation template is part of the data interface it learned.

## The objective and its exact weighting

For example $i$ with prompt $x_i$ and response $y_i$, including EOS:

$$
\ell_i=-\sum_{t=1}^{M_i}\log p_\theta(y_{it}\mid x_i,y_{i,<t}).
$$

The batch loss is $\sum_i\ell_i/\sum_iM_i$, the average over supervised tokens.
Prompt labels and padding labels are -100, so they do not enter this sum.
The model still conditions on the prompt. Gradients can flow through prompt
representations because they affect response predictions.

`model(**batch).loss` performs the causal shift internally. `collate` therefore
passes unshifted full-sequence labels. The first answer label aligns with the
logit computed at the last prompt position after the model's internal shift.

The script encodes the prompt and the response separately, then concatenates
the resulting IDs. This defines a clear token boundary for the label mask;
it need not equal tokenizing the concatenated string with every tokenizer.
Use and verify the model's own template/masking procedure for a real chat corpus.

## Padding and EOS can share an ID without sharing a role

For GPT-2 tokenizers, padding is often assigned the EOS ID. Consider a sequence
whose final real token is EOS and which needs one additional pad position.
The two IDs are identical, but the real EOS must be predicted and the pad ignored.

That is why the collator uses lengths to construct masks:

```python
attention_mask = [1] * real_length + [0] * padding_length
labels = real_labels + [-100] * padding_length
```

Masking every token equal to the padding ID would incorrectly remove genuine
EOS supervision. The batch's attention mask and labels have separate purposes.

## Full tuning, LoRA, and QLoRA

Full tuning directly changes the original parameters $\theta$.
LoRA substitutes updates $W=W_0+(\alpha/r)BA$ and changes only the adapter
parameters. Read [the LoRA derivation](03_lora_from_scratch.md) first.

The optional QLoRA path stores a quantized frozen base, while training
floating-point adapters. Conceptually:

$$
W_{eff}\approx\operatorname{dequantize}(Q(W_0))+sBA.
$$

A quantizer stores a small code plus scaling information; its reconstruction
is an approximation. NF4 uses a nonuniform 4-bit codebook suited to roughly
normal weight distributions. Double quantization also compresses quantization
constants. The code delegates those details to bitsandbytes; it does not
implement NF4 arithmetic itself. Training still needs activations, temporary
computation, and adapter optimizer state. Four-bit storage is not four-bit
end-to-end backpropagation. See the [official quantization guide](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

## Follow one update through the script

1. Load and validate nonempty JSONL data; reject exact overlapping prompts.
2. Encode examples and reject overlong ones rather than silently cutting answers.
3. Collate two examples into a padded tensor batch.
4. Clear gradients, run the causal model, and compute masked token loss.
5. Backpropagate, clip gradient norm to 1, and apply AdamW.
6. Repeat until the requested number of updates has run, cycling the data loader.

The learning rate is $2\times10^{-4}$ for LoRA and $2\times10^{-5}$ for full
tuning. Those are example settings, not guaranteed suitable rates. `steps` counts
optimizer updates, not epochs. With eight examples and batch size two, four
updates traverse the dataset once.

Validation is aggregated by supervised-token count. If batch A has 10 tokens
and mean loss 2, and batch B has 30 tokens and mean loss 4, the correct combined
mean is $(10\cdot2+30\cdot4)/40=3.5$, not 3. The script does exactly this.
Checking exact prompt overlap catches one leak type, not paraphrase or source
leakage; real data needs stronger provenance-aware splits.

## Run and interpret

Install the optional requirements, then try:

```bash
python 02_finetuning/04_huggingface_sft.py --model sshleifer/tiny-gpt2 --steps 5
```

This tiny model tests plumbing. Its generated text may be repetitive and its
validation loss may not improve. Save paths contain either full model weights
or adapter weights, a tokenizer, and a run configuration. The saved configuration
does not contain optimizer state, so it is for inference and documentation,
not exact continuation of an interrupted training run.

Exercise: print the decoded supervised tokens for one batch; verify that they
contain the response and EOS only. Compare full and LoRA training on a larger
held-out task before making quality claims. Continue to [reload and generation](05_reload_and_generate.md).
