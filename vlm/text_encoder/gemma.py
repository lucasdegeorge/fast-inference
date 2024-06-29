from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from llm.architecture.base import LLMArgs, Attention
from .base import TransformerForVLM, VLMArgs
from llm.architecture.gemma import FeedForwardForGemma, RMSNormForGemma

## Mostly adapted from llm/architecture/base.py
## and vlm/text_encoder/base.py


class TransformerForGemmaVLM(TransformerForVLM):
    def __init__(self, config: VLMArgs) -> None:
        super().__init__(config)

        ## Lines with new classes for Gemmma VLM models
        self.layers = nn.ModuleList(
            TransformerBlockForGemmaVLM(config.llm_args)
            for _ in range(config.llm_args.n_layers)
        )
        self.norm = RMSNormForGemma(config.llm_args.dim, eps=config.llm_args.norm_eps)

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

        ## Line for Gemma models
        x = (self.config.llm_args.dim**0.5) * x

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


class TransformerBlockForGemmaVLM(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForwardForGemma(
            config
        )  ## FeedForwardForGemma instead of FeedForward
        self.ffn_norm = RMSNormForGemma(
            config.dim, config.norm_eps
        )  ## RMSNormForGemma instead of RMSNorm
        self.attention_norm = RMSNormForGemma(
            config.dim, config.norm_eps
        )  ## RMSNormForGemma instead of RMSNorm

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
