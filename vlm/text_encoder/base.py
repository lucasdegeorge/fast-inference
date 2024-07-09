from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass

from vlm.utils import vlm_configs, vlm_configs
from llm.architecture.base import (
    LLMArgs,
    Transformer,
    Attention,
    FeedForward,
    RMSNorm,
    KVCache,
    precompute_freqs_cis,
    find_multiple,
)

## Mostly adapted from llm/architecture/base.py


@dataclass
class VLMArgs:
    llm_args: LLMArgs
    hf_checkpoint: str = None
    image_token_index: int = 257152
    num_image_tokens: int = 256
    # TODO: Add more args for non-HF implementations

    @classmethod
    def from_name(cls, name: str):
        if name in vlm_configs:
            return cls(**vlm_configs[name])
        else:
            raise ValueError(f"Unknown VLM model name: {name}")


class TransformerForVLM(Transformer):
    def __init__(self, config: VLMArgs) -> None:
        super().__init__(config.llm_args)
        self.config = config

        ## Lines with new classes for VLM models
        self.layers = nn.ModuleList(
            TransformerBlockForVLM(config.llm_args)
            for _ in range(config.llm_args.n_layers)
        )

    def setup_caches(self, max_batch_size, max_seq_length, device: torch.device):
        """Similar to the setup_caches method in llm/architecture/base.py with config replaced by config.llm_args."""
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.llm_args.n_local_heads,
                self.config.llm_args.head_dim,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.llm_args.block_size,
            self.config.llm_args.head_dim,
            self.config.llm_args.rope_base,
            dtype,
            device,
        )
        self.causal_mask = torch.tril(
            torch.ones(
                self.max_seq_length,
                self.max_seq_length,
                dtype=torch.bool,
                device=device,
            )
        )

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        embeds: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        ## New line for VLM models
        if embeds is not None:
            x = embeds
        else:
            x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(VLMArgs.from_name(name))


class TransformerBlockForVLM(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        ## New line for VLM models
        if mask.shape[2] > 1:
            inp_size = mask.shape[2]
            mask[:, :, :inp_size, :inp_size] = torch.ones_like(
                mask[:, :, :inp_size, :inp_size]
            )

        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
