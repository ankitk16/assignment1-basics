import random

import numpy as np
import torch
from einops import einsum, rearrange, reduce
from torch import nn


class Linear(nn.Module):
    """
    Linear moudle where it appears with weight, say W and input x is doing xW. This is because the shape of x
    is ... d_in and shape of W is d_in, d_out. And the shape of output is ... d_out. That happens with xW not Wx, even
    though the notations in document says Wx. To compensate for all these the weights are saved in the shape d_out,
    d_in. Although, while the linear module constructor takes argument model_in first and then model_out.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        std = (2.0 / (in_features + out_features)) ** 0.5
        w = nn.init.trunc_normal_(
            w,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return x


class Embedding(nn.Module):
    """
    (num_embeddings) is vocab size and (embedding_dim) is (d_model).
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        w = nn.init.trunc_normal_(w, 0.0, 1.0, a=-3, b=3)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.long)
        x = self.weight[x]
        return x


class RMSNorm(nn.Module):
    """
    The role of this class is that it does across channel normalization of submitted batch of examples. The
    normalization is with respect to SD. There is no mean normalization.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        w = torch.empty(d_model, device=device, dtype=dtype)
        # w_norm = torch.linalg.vector_norm(w)
        w = nn.init.trunc_normal_(w, std=1.0)
        self.weight = nn.Parameter(w)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # upcast your input to torch.float32 to prevent overflow when you square the input
        x = x.to(torch.float32)
        x_channel_sq = x**2  # point-wise squre
        x_channel_sq = reduce(x_channel_sq, "... d_model -> ... 1", "mean")
        x_channel_sq = x_channel_sq**0.5  # point-wise squre root
        x = x / x_channel_sq  # point-wise division by broadcasting last
        x = x * self.weight
        x = x.to(in_dtype)
        return x


class SwiGLU(nn.Module):
    """
    Questions:
    1. Initializtion mean and std might need more thinking.
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # Linear is by in_features, out_features
        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)  # d_ff
        return self.w2(a * torch.sigmoid(a) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """
    We are going to receive input q with dim(d_k). We will modify it and return output with the same dim.
    if d_k_i odd:
        d_k_i_out = d_k_i * cos(thi) - q_k_(i+1) * sin (thi)
    else:
        d_k_i_out = d_k_(i-1) * sin(thi) + q_k_(i) * cos (thi)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for pair wise rotation"

        # inverse frequencies, one per pair: theta^(-2k/d_k) for k = 0 .. d_k/2 - 1
        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = 1.0 / (theta**k)  # dim = d_k /2

        # dim = max_seq_len
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angels = einsum(pos, inv_freq, "max_seq_len, d_k_2 -> max_seq_len d_k_2")

        self.register_buffer("cos", torch.cos(angels), persistent=False)
        self.register_buffer("sin", torch.sin(angels), persistent=False)

    def forward(self, x: torch.Tensor, token_positions=torch.Tensor) -> torch.Tensor:
        # x has dim ... max_seq_len d_k
        # token_postions has dim (..., max_seq_len)
        cos = self.cos[token_positions]  # dim ..., max_seq_len, d_k/2
        sin = self.sin[token_positions]

        # x_pair will have dim ... max_seq_len d_k/2 2
        x_pairs = rearrange(x, "... (half two) -> ... half two", two=2)
        x1 = x_pairs[..., 0]  # odd channels; dim ... seq_len d_k/2
        x2 = x_pairs[..., 1]  # even channels

        # 3. Rotate each pair.
        out1 = x1 * cos - x2 * sin  # both has dim ... seq_len d_k/2
        out2 = x1 * sin + x2 * cos  # both has dim ... seq_len d_k/2
        # stack creates the new axis but doesn't collapse it
        out = torch.stack((out1, out2), dim=-1)  # dim ... seq_len d_k/2, 2
        # now collapse the last axis
        return rearrange(out, "... half two -> ... (half two)").to(x.dtype)


def scaled_dot_product_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None,
):
    """
    Mask is always has dimensions (seq, seq).
    """
    d_k = query.shape[-1]
    qk = einsum(query, keys, "... seq_a d_k, ... seq_b d_k -> ... seq_a seq_b") / (
        d_k**0.5
    )  # this is often referred to as scores

    if mask is not None:  # mask is seq*seq dim
        qk = qk.masked_fill(mask == False, float("-inf"))
        qk = softmax(qk, dim=-1)

    return einsum(qk, values, "... seq_a seq_b, ... seq_b d_v -> ... seq_a d_v")


def softmax(x: torch.Tensor, dim: int):
    # does softmax along the dimenation given
    largest = x.amax(dim, keepdim=True)
    x_normalized = x - largest
    x_normalized = torch.exp(x_normalized)
    x_normalized_sum = x_normalized.sum(
        dim, keepdim=True
    )  # reduce(x_normalized, "... dim -> ... 1", "sum")
    x = x_normalized / x_normalized_sum
    return x


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1, logits.shape[-1])  # (N, V) for any input rank
    targets = targets.reshape(-1)  # (N,)

    m = logits.amax(-1, keepdim=True)
    z = logits - m
    lse = torch.log(torch.exp(z).sum(-1))  # (N,)
    picked = z.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    return (lse - picked).mean()


def learning_rate_schedule(t: int, alpha_max, alpha_min, tw, tc):
    if t < tw:
        return t * alpha_max / tw
    elif t > tc:
        return alpha_min
    else:
        return alpha_min + 0.5 * (1 + math.cos((t - tw) * torch.pi / (tc - tw))) * (
            alpha_max - alpha_min
        )


def gradient_clipping(params: list[torch.Tensor], norm_max: float, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if total_norm > norm_max:
        scale = norm_max / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(x, batch_size: int, context_length: int, device: str):
    n = len(x)

    last_permissable = n - context_length - 1
    starts = [random.randint(0, last_permissable) for _ in range(batch_size)]
    inputs = np.stack([x[s : s + context_length] for s in starts])
    outputs = np.stack([x[s + 1 : s + context_length + 1] for s in starts])
    to = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)

    return to(inputs), to(outputs)


import os
import typing


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {}
    # state_dict returns dict with model weights/buffers' names as keys and their values as values
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration
    torch.save(obj=obj, f=out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # we have saved a dict objects. so load will create the same obj
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


class Multi_Head_Self_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_k = self.d_v = d_model // num_heads
        self.h = num_heads
        self.k_proj = Linear(
            out_features=self.h * self.d_k,
            in_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.q_proj = Linear(
            out_features=self.h * self.d_k,
            in_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            out_features=self.h * self.d_v,
            in_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            in_features=self.h * self.d_v,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
            if theta is not None
            else None
        )
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        """
        What kind of mask we need?
        """
        seq = x.shape[-2]
        # end shape is selected such that each head is seperately processed
        # moving from gian seq*(h*dk) matrix to h matrices of shape (seq*dk)
        q = rearrange(self.q_proj(x), "... seq (h dk) -> ... h seq dk", h=self.h)
        k = rearrange(self.k_proj(x), "... seq (h dk) -> ... h seq dk", h=self.h)
        v = rearrange(self.v_proj(x), "... seq (h dv) -> ... h seq dv", h=self.h)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # lower triang matrix for mask with dim seq
        mask = torch.ones(seq, seq, device=x.device)
        mask = torch.tril(mask)

        # let's get attention; shape is ... h seq d_v
        attn_sepearte = scaled_dot_product_attention(q, k, v, mask)
        # concacte all the heads back =>  ... seq (h*d_v)
        attn_concated = rearrange(attn_sepearte, "... h seq d_v -> ... seq (h d_v)")
        return self.output_proj(attn_concated)


class Transformer_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.attn = Multi_Head_Self_Attention(
            d_model, num_heads, theta, max_seq_len, device, dtype
        )
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = self.ln1(x)
        casual_attn = self.attn(x_normed)
        x = x + casual_attn
        x_normed = self.ln2(x)
        x_ff = self.ffn(x_normed)
        return x + x_ff


class Transformer(nn.Module):
    """
    Linear moudle where it appears with weight, say W and input x is doing xW. This is because the shape of x
    is ... d_in and shape of W is d_in, d_out. And the shape of output is ... d_out. That happens with xW not Wx, even
    though the notations in document says Wx. To compensate for all these the weights are saved in the shape d_out, d_in.

    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        num_layers: int,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        block_parameters = (
            d_model,
            num_heads,
            d_ff,
            theta,
            context_length,
            device,
            dtype,
        )
        self.layers = nn.ModuleList(
            Transformer_Block(*block_parameters) for _ in range(num_layers)
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        # last norm in the figure
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # head is just before soft max (Linear in Figure)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)  # RMS Norm
        x = self.lm_head(x)  # convert to Logits via head
        return x


# __________SGD Code Given in the assignment______#
import math
from collections.abc import Callable

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, weight_decay: float, betas=(0.9, 0.999), lr=1e-3, eps=1e-8
    ):
        if lr < 0:
            raise ValueError("lr<0")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # decorator
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            decay = group["weight_decay"]
            beta = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # first get state
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                alpha_t = lr * (((1 - beta[1] ** t) ** 0.5) / (1 - beta[0] ** t))
                # update parameter
                p.mul_(1 - lr * decay)
                m = m * beta[0] + (1 - beta[0]) * g
                v = v * beta[1] + (1 - beta[1]) * (g**2)
                p.data -= alpha_t * m / ((v.sqrt()) + eps)
                # update state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


# _______Activation Storage____#
def activation_storage():
    """
    Storing intermediate stuff in forward pass till reaching losses since we need them to calculate loss.backward().
    Once backward is calculated they are thrown away by PyTorch autograd.
    In each layer (all scales with # examples in a batch):
        1. RMS Norm Outputs: L * d_model. Since for each token in context, mean is taken of all the features.
        2. Q, K, V (e.g., V= W_v @ x): 3 * L * d_model. Note L * d_model is dimention of V.  In other words,
            we are storing outputs of W_v @ x.
        3. Q(K^T) which is often referred to as scores takes h* L^2. This happens inside scaled_dot_product_attn_fun.
        4. Next softmax to above is applied to mask j>i that takes again h*L^2 since pointwise.
        5. Next multiply the last one by V. And then that with W_output. The output requires storage of 2* L * d_model
            since each is  L * d_model.
        6. SwiGLU: W1 and W3 outputs plus the product, 3 · L · d_ff. The factor 3 instead of 2 since there is
          pointwise mul of W1 and W3 as well.  W2 output L · d_model
    Once at the end: embeddings L · d_model, final norm L · d_model, logits L · vocab_size, and softmax over them.
    Multiply the per-layer sum by num_layers, add the one-off terms, multiply by batch_size, and by 4 for fp32.
    """
    return
