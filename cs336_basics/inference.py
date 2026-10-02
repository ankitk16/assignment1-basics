import torch
from torch import nn

from cs336_basics.nn_modules import (
    softmax,
)


def top_p_filter(probs: torch.Tensor, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask out token once cum prob reaches threshold
    sorted_mask = (cumulative_probs - sorted_probs) > p
    sorted_probs[sorted_mask] = 0.0

    # scatter back to orignal vocab ordering
    probs = torch.zeros_like(probs).scatter_(
        dim=-1, index=sorted_indices, src=sorted_probs
    )
    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
    return probs


@torch.no_grad()
def gen_text(
    prompt: torch.Tensor,
    model: nn.Module,
    n_tok: int,
    eot: int,
    context_length: int,
    temp: float | None = None,
    top: float | None = None,
):
    """
    Takes x which is prompt of single dimension.
    """
    x = prompt
    tok_gen = []

    for _ in range(n_tok):
        ctx = x[-context_length:]
        logits = model(ctx.unsqueeze(0))[0, -1, :]  # unsqueeze to give a batch dim
        assert logits.dim() == 1, logits.shape

        if temp:
            logits = logits / temp

        probs = softmax(logits, dim=-1)

        if top:
            probs = top_p_filter(probs, p=top)

        # now generate the next token
        ix = torch.multinomial(probs, num_samples=1)
        tok_gen.append(ix.item())
        # now update the context token
        x = torch.cat([x, ix])
        if tok_gen[-1] == eot:
            break

    return tok_gen


# Take stock of situation
# 1. I am cold
# 2. Thinking needs a better space
# 3. I am hungry so I need some food--I could eat at home and then go to Nero?
# 4.
