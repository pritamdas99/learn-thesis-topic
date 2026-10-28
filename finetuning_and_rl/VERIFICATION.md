# Verification record

Verified on 2026-09-23 using Python 3.12.3 on CPU. Packages: PyTorch
2.14.0+cpu, NumPy 2.5.3, Transformers 5.17.0, PEFT 0.21.0, Accelerate 1.15.0,
and Safetensors 0.8.0. See `requirements-tested.txt` for top-level version pins.
The broader requirements ranges are convenience ranges, not a guarantee that
every possible combination has been tested.

## Executed successfully

- All 27 lesson scripts passed Python syntax compilation.
- `python run_lessons.py`: 24/24 offline single-process lessons completed.
- `python -m unittest discover -s tests -v`: 13/13 numerical checks passed.
- Two CPU processes ran the DDP lesson successfully. Learned regression weights
  were approximately [0.999, 0.999, 1.001, 0.997], against a target of all ones.
- The real-model SFT script loaded `sshleifer/tiny-gpt2`, ran five LoRA updates,
  evaluated the validation split, and saved the adapter and tokenizer.
- The saved LoRA adapter was reloaded with its original base and generated text.
- The full fine-tuning branch ran two updates, saved a full model checkpoint,
  and successfully reloaded it for generation.

The numerical tests cover future-token leakage, prompt/padding supervision,
LoRA initialization and merging, Bellman values, discounted returns, terminal
versus truncated GAE behavior, DPO gradient direction, masked sequence
log-probabilities, constant-reward GRPO, conservative Q gradients, and preserving
real EOS targets when EOS is also the padding token.

## Limits of this verification

These are execution and mathematical-invariant checks, not benchmark claims.
The tiny GPT-2 model produced repetitive text and its validation loss did not
improve in the short test. That is expected for a plumbing check and is not a
useful assistant. The larger default `distilgpt2` model was not downloaded/tested.

CUDA mixed precision, NVIDIA QLoRA, large models, multi-GPU training, and other
package versions were not tested. SAC/CQL and the categorical alignment labs
teach objectives; they are not complete large-scale training systems.

DDP initially could not open local sockets in the restricted execution sandbox;
it passed when run with local socket access. Package/model downloads likewise
needed network access. Core lessons run offline after installing dependencies.

## Theory companion update

Added a matching `.md` chapter for each of the 27 lesson scripts, plus a math
primer and a paired theory/code index. This update changes documentation only.

Checked all 27 companion files exist, all local Markdown links resolve, and
fenced code/display-equation delimiters are balanced. Independently recomputed
the worked GAE recurrence and return targets, PPO clipping table, finite-difference
gradient, broadcasting-error loss, categorical RLHF optimum, and DPO example.
These checks passed. They are arithmetic/document-structure checks, not a claim
that every Markdown viewer renders LaTeX identically.

The SAC companion explicitly explains the existing example's identical critic
initialization, and the policy-gradient companion distinguishes its simplified
finite-horizon and discount-weighting conventions from a general exact objective.

## Requested RL syllabus update — 2026-10-22

Verified using Python 3.12.3, PyTorch 2.14.1+cpu, and NumPy 2.5.2 in
`/tmp/thesis-rl-venv` (temporary verification environment).

- All 31 lesson scripts passed Python syntax parsing; all local Markdown links resolved.
- Offline lesson runner: **28/28 passed**; standard-library subset: **6/6 passed**.
- Numerical invariant suite: **17/17 passed**. Added checks for policy-iteration
  agreement, actual TRPO KL and improvement, masked distillation/teacher stop-gradient,
  and RLSD sign preservation and zero-mixing behavior.
- Original DQN (`--vanilla-dqn`) also ran successfully, selecting right in all
  three nonterminal states. The default Double DQN passed in the full runner.
- A2C and concurrent A3C trained successfully. A3C scheduling is nondeterministic;
  observed right probabilities were approximately [0.922, 0.945, 0.912].
- TRPO's bandit lab reached approximately 0.993 probability on the best action.
- SDPO, OPSD, and JSD objective losses decreased, with no teacher gradients.

These checks exercise teaching examples, not published LLM results. The TRPO
and self-distillation additions are objective/update labs; they do not implement
complete trajectory-based TRPO or real language-model post-training pipelines.
The existing external-download and DDP scripts were not rerun for this update.
