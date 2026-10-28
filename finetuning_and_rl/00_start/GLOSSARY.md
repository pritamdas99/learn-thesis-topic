# Vocabulary beside the code

| Term | Meaning |
| --- | --- |
| Parameter / weight | A learned number inside a model |
| Hyperparameter | A choice such as learning rate, rank or batch size |
| Tensor | A multidimensional array with shape and numeric type |
| Logit | Unnormalized score before softmax |
| Softmax | Converts scores into probabilities summing to one |
| Log-probability | Log of a probability; products become sums |
| Cross-entropy | Negative log-probability assigned to the target |
| Gradient | Derivatives of an objective with respect to parameters |
| Autograd | Automatic differentiation using the chain rule |
| Backpropagation | Reverse traversal that computes those derivatives |
| Optimizer | A rule for updating parameters from gradients |
| AdamW | Adaptive gradient optimizer with decoupled weight decay |
| Epoch / step | Pass through data / a training update (check local meaning) |
| Generalization | Performance on unseen data from the intended task |
| Leakage | Evaluation information accidentally entering training/selection |
| Embedding | Learned vector representing a token or other discrete item |
| Attention | Content-dependent weighted aggregation of representations |
| Causal mask | Prevents attending to future positions |
| Pretraining | Initial broad learning, often next-token prediction |
| SFT | Supervised fine-tuning on desired examples |
| PEFT / LoRA / QLoRA | Parameter-efficient tuning / low-rank updates / quantized-base LoRA |
| Policy pi(a given s) | Distribution of actions given state/observation |
| State / observation | Environment's underlying condition / what the agent sees |
| Reward r | Immediate feedback after an action |
| Return G | Sum of future rewards, usually discounted |
| Discount gamma | Weight on future rewards; often between zero and one |
| V(s) | Expected return from a state under a policy |
| Q(s,a) | Expected return after a state/action pair under a policy |
| Advantage A(s,a) | Q(s,a)-V(s): better/worse than the state baseline |
| TD error | Sampled reward plus bootstrap value minus current estimate |
| Bootstrap | Use a learned future estimate in a training target |
| On-policy | Learn from the policy currently being improved |
| Off-policy | Learn a target policy using behavior from another policy |
| Replay buffer | Stored transitions reused for learning |
| Actor / critic | Policy model / value estimator |
| Entropy | Uncertainty of an action distribution |
| KL divergence | Directional measure of difference between distributions |
| GAE | Smoothed sum of temporal-difference errors estimating advantage |
| RLHF / RLAIF | Reinforcement learning using human / AI-derived feedback |
| DPO | Direct preference optimization from chosen/rejected examples |
| GRPO | Group-relative policy optimization using within-prompt rewards |
| Quantization | Represent numbers using fewer bits and scaling/codebooks |
| Distillation | Train a student to approximate a teacher's behavior |
| Ablation | Remove/change one component to test its contribution |

Notation: B=batch size, T=sequence length, V=vocabulary size, d=hidden width,
r=LoRA rank, alpha=adapter scale (or an unrelated SAC temperature),
beta=reference regularization scale. Symbols can mean different things in
different algorithms; read each local definition.
