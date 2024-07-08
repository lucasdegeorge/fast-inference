import torch
from torch import Tensor
from typing import Optional


transformer_configs = {
    "Llama-2-7B-chat-hf": dict(
        checkpoint_path="checkpoints/meta-llama/Llama-2-7B-chat-hf",
        n_layers=32,
        n_heads=32,
        dim=4096,
    ),
    "Llama-2-13B-chat-hf": dict(
        checkpoint_path="checkpoints/meta-llama/Llama-2-13b-chat-hf",
        n_layers=40,
        n_heads=40,
        dim=5120,
    ),
    "Mistral-7B": dict(
        checkpoint_path="checkpoints/mistralai/Mistral-7B-Instruct-v0.2",
        n_layers=32,
        n_heads=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "Meta-Llama-3-8B": dict(
        checkpoint_path="checkpoints/meta-llama/Meta-Llama-3-8B",
        block_size=8192,
        n_layers=32,
        n_heads=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
    ),
    "Meta-Llama-3-8B-Instruct": dict(
        checkpoint_path="/home/lucas/Documents/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct",
        block_size=8192,
        n_layers=32,
        n_heads=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
    ),
    "TinyLlama-1.1B-Chat-v1.0": dict(
        checkpoint_path="checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_layers=22,
        n_heads=32,
        dim=2048,
        intermediate_size=5632,
        vocab_size=32000,
    ),
    "TinyLlama-1.1B-intermediate-step-1431k-3T": dict(
        checkpoint_path="checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        block_size=2048,
        vocab_size=32000,
        intermediate_size=5632,
        n_layers=22,
        n_heads=32,
        n_local_heads=4,
        dim=2048,
    ),
    "Mistral-7B-Instruct-v0.2": dict(
        checkpoint_path="checkpoints/mistralai/Mistral-7B-Instruct-v0.2",
        n_layers=32,
        n_heads=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "gemma-7b-it": dict(
        checkpoint_path="checkpoints/google/gemma-7b-it",
        dim=3072,
        vocab_size=256000,
        n_layers=28,
        n_heads=16,
        n_local_heads=16,
        intermediate_size=24576,
        head_dim=256,
    ),
    "gemma-2b-it": dict(
        checkpoint_path="checkpoints/google/gemma-2b-it",
        dim=2048,
        block_size=8192,
        vocab_size=256000,
        n_layers=18,
        n_heads=8,
        n_local_heads=1,
        intermediate_size=16384,
    ),
    "paligemma-3b-mix-224": dict(
        checkpoint_path="checkpoints/google/paligemma-3b-mix-224",
        block_size=8192,
        dim=2048,
        vocab_size=257216,
        n_layers=18,
        n_heads=8,
        n_local_heads=1,
        intermediate_size=16384,
        rope_base=10000,
    ),
}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype, device=device)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")
