"""SDPO, OPSD and RLSD objective labs, not complete LLM trainers.
Shapes: logits [batch, time, vocabulary], token IDs and masks [batch, time].
Read the companion chapter for the chosen paper versions and simplifications.
"""
import torch


def masked_mean(values, mask):
    # Average tokens within each response, then average valid responses.
    counts = mask.sum(-1)
    if not (counts > 0).all():
        raise ValueError('Each response must contain at least one valid token')
    return ((values*mask).sum(-1)/counts).mean()


def distillation_loss(student_logits, teacher_logits, mask, kind='sdpo'):
    student = student_logits.log_softmax(-1)
    teacher = teacher_logits.detach().log_softmax(-1)
    if kind == 'sdpo':  # reverse KL: student || feedback-conditioned teacher
        token_loss = (student.exp()*(student-teacher)).sum(-1)
    elif kind == 'opsd':  # forward-KL teaching choice: teacher || student
        token_loss = (teacher.exp()*(teacher-student)).sum(-1)
    elif kind == 'jsd':
        mixture = torch.logaddexp(student, teacher)-torch.log(torch.tensor(2.))
        token_loss = 0.5*((teacher.exp()*(teacher-mixture)).sum(-1)
                          +(student.exp()*(student-mixture)).sum(-1))
    else:
        raise ValueError(kind)
    return masked_mean(token_loss, mask)


def rlsd_advantages(sequence_advantages, teacher_token_logs, old_token_logs,
                    mixing=0.5, weight_clip=0.2):
    if not 0 <= mixing <= 1 or not 0 <= weight_clip < 1:
        raise ValueError('mixing in [0,1], weight_clip in [0,1) required')
    with torch.no_grad():
        gap = teacher_token_logs-old_token_logs
        signed_gap = sequence_advantages[:, None].sign()*gap
        weights = signed_gap.clamp(-30, 30).exp().clamp(1-weight_clip, 1+weight_clip)
        return sequence_advantages[:, None]*((1-mixing)+mixing*weights)


def rlsd_loss(new_token_logs, old_token_logs, teacher_token_logs,
              sequence_advantages, mask, mixing=0.5, policy_clip=0.2):
    advantages = rlsd_advantages(sequence_advantages, teacher_token_logs,
                                 old_token_logs, mixing)
    ratio = (new_token_logs-old_token_logs.detach()).exp()
    objective = torch.minimum(ratio*advantages,
                              ratio.clamp(1-policy_clip, 1+policy_clip)*advantages)
    return -masked_mean(objective, mask)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    teacher = torch.randn(2, 3, 4, requires_grad=True)
    for kind in ('sdpo', 'opsd', 'jsd'):
        student = torch.zeros(2, 3, 4, requires_grad=True)
        optimizer = torch.optim.SGD([student], lr=0.5)
        before = distillation_loss(student, teacher, mask, kind).item()
        for _ in range(50):
            optimizer.zero_grad()
            loss = distillation_loss(student, teacher, mask, kind)
            loss.backward()
            optimizer.step()
        after = distillation_loss(student, teacher, mask, kind).item()
        print(kind, 'loss:', before, '->', after)
        assert after < before and teacher.grad is None
    # Actions sampled from the unprivileged old student, not the teacher.
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    with torch.no_grad():
        old_all = logits.log_softmax(-1)
        tokens = torch.distributions.Categorical(logits=logits).sample()
        old = old_all.gather(-1, tokens[..., None]).squeeze(-1)
        teacher_logs = teacher.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    new = logits.log_softmax(-1).gather(-1, tokens[..., None]).squeeze(-1)
    loss = rlsd_loss(new, old, teacher_logs, torch.tensor([1., -1.]), mask)
    loss.backward()
    print('RLSD sampled-token loss:', loss.item())
    assert torch.isfinite(logits.grad).all() and teacher.grad is None


if __name__ == '__main__':
    main()
