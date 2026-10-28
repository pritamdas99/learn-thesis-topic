"""Direct Preference Optimization without an online reward model.
DPO uses fixed prompt/chosen/rejected triples and a frozen reference policy.
loss = -log sigmoid(beta * [(log pi_w-log ref_w)-(log pi_l-log ref_l)]).
DPO is offline preference optimization; it does not collect RL rollouts.
This categorical example trains end-to-end; sequence_logprob shows the LLM bridge.
Exercise: make beta smaller and compare gradient size and final preference.
"""
import torch
from torch import nn


def sequence_logprob(logits, token_ids, response_mask):
    # [B,T,V] logits; [B,T] IDs/mask. Mask marks REAL response tokens, not pads.
    token_logs = logits[:, :-1].log_softmax(-1).gather(
        -1, token_ids[:, 1:, None]).squeeze(-1)
    # Original DPO uses SUMMED completion log-probabilities, not token means.
    return (token_logs * response_mask[:, 1:]).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (policy_chosen-policy_rejected)-(ref_chosen-ref_rejected)
    return -nn.functional.logsigmoid(beta*margin).mean()


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    logits = nn.Parameter(torch.zeros(3))
    reference_logs = torch.zeros(3).log_softmax(-1)  # Frozen uniform reference.
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(100):
        logs = logits.log_softmax(-1)
        loss = dpo_loss(logs[2], logs[0], reference_logs[2], reference_logs[0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('probabilities [rejected, neutral, chosen]:', logits.softmax(-1).detach().tolist())
    # For LLMs, chosen and rejected must use the SAME prompt/template and the
    # reference must not be updated. Padding/prompt labels must be excluded.

if __name__ == '__main__':
    main()
