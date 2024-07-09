from typing import Optional
import torch
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.utils import logits_to_probs, multinomial_sample_one_no_sync
from llm.architecture.base import LLMArgs

vlm_configs = {
    "paligemma-3b-mix-224": dict(
        llm_args=LLMArgs.from_name("paligemma-3b-mix-224"),
        image_token_index=257152,
        num_image_tokens=256,
        hf_checkpoint="google/paligemma-3b-mix-224",
    ),
}


def sample_vlm(logits, temperature: float = 0.8, top_k: int = 200):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def to_rgb(img):
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img
