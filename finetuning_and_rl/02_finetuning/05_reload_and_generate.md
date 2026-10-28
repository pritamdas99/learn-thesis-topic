# Reloading models and generating tokens

Companion code: [05_reload_and_generate.py](05_reload_and_generate.py). Read this explanation before running the code.

## What has to be saved to reproduce behavior?

A language model's behavior depends on its architecture, learned weights,
tokenizer, input format, and generation rule. Weights without their compatible
tokenizer can turn text into the wrong IDs. A LoRA adapter additionally needs
the original base model from which it was trained.

Read [the SFT workflow](04_huggingface_sft.md). This script performs inference,
not optimization: there is no loss, backward pass, or optimizer update.

## Full checkpoint versus adapter checkpoint

A full checkpoint represents the tuned weights $W_{full}$. An adapter checkpoint
represents the update factors, conceptually $A,B$ in:

$$
W_{effective}=W_{base}+(\alpha/r)BA.
$$

The presence of `adapter_config.json` selects the adapter-loading branch.
`PeftConfig` identifies the compatible base model; the base is loaded, then
`PeftModel.from_pretrained` attaches the saved adapters. Otherwise,
`AutoModelForCausalLM.from_pretrained` loads the full checkpoint directly.

The tokenizer is loaded from the output directory in both cases. For strict
reproducibility, also pin base model revisions, package versions, and template
behavior rather than relying only on a mutable repository name.

## Autoregressive generation, mathematically

For prompt tokens $x$ and generated tokens $y$:

$$
p(y\mid x)=\prod_{t=1}^M p(y_t\mid x,y_{<t}).
$$

At every step the model produces vocabulary logits. Greedy decoding chooses:

$$
y_t=\arg\max_k z_{t,k}.
$$

The script uses `do_sample=False`, so generation is greedy. Greedy token choices
do not necessarily maximize the probability of the whole completed sequence.
For example, a first token with probability 0.6 followed by a best continuation
of probability 0.1 gives joint probability 0.06, whereas another first token
with probability 0.4 and continuation probability 0.9 gives 0.36.
Greedy decoding picks the first option without exploring that future.

Sampling instead draws from $\operatorname{softmax}(z/\tau)$, with temperature
$\tau$. Top-p sampling keeps a prefix of probability-ranked tokens whose mass
reaches a chosen threshold, renormalizes, and samples. These alter decoding;
they do not update or repair the trained model.

## Connect the code to behavior

| Code | Purpose |
| --- | --- |
| `model.eval()` | Disable training-specific behavior such as dropout |
| `with torch.no_grad()` | Avoid recording a backward graph |
| `'Question: '+prompt+'\nAnswer:'` | Match the format used by the SFT lesson |
| `max_new_tokens=40` | Limit additional tokens, excluding the original prompt |
| `output[0, input_length:]` | Decode only the new suffix, not prompt plus answer |
| `skip_special_tokens=True` | Hide tokens such as EOS in the displayed text |

Generation can stop before 40 tokens if EOS is produced. The CPU reload path
loads an unquantized base even for a QLoRA-trained adapter; this can require
substantially more RAM than the quantized training base. Other hardware/loading
configurations need explicit setup.

## Run and check

After running the SFT lesson, execute
`python 02_finetuning/05_reload_and_generate.py`.
Use `--checkpoint outputs/full` for the full-model example. Expect a string of
new tokens, not necessarily a correct answer from a tiny teaching model.

Exercise: compare greedy and sampled output using the same saved model. Explain
which source of variation changed: decoding randomness, not model parameters.
For a stricter save/reload check, compare evaluation-mode logits on fixed input
IDs before saving and after loading, allowing for floating-point tolerances.
