from pathlib import Path
import torch
from llm.model import LLMBase, GemmaBase
from vlm.model import VLMBase, GemmaVLMBase
import argparse


def main(
    is_vlm: bool,
    model_name: str,
    prompt: str,
    image: str,
    compile: bool,
    quant: str,
    nb_samples: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
):
    if is_vlm:
        if "gemma" in model_name.lower():
            vlm = GemmaVLMBase(model_name, compile=compile, quant=quant)
        else:
            vlm = VLMBase(model_name, compile=compile, quant=quant)
        vlm.benchmark_tok_per_s(
            prompt,
            image,
            max_new_tokens,
            nb_samples,
            top_k=top_k,
            temperature=temperature,
        )
    else:
        if "gemma" in model_name.lower():
            llm = GemmaBase(model_name, compile=compile, quant=quant)
        else:
            llm = LLMBase(model_name, compile=compile, quant=quant)
        llm.benchmark_tok_per_s(
            prompt, max_new_tokens, nb_samples, top_k=top_k, temperature=temperature
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_vlm", action="store_true", help="Whether to use VLM model or LLM model"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="tennis_player.jpeg",
        help="Image path (for VLM only).",
    )
    parser.add_argument("--nb_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Meta-Llama-3-8B",
        help="Model name (see README.md).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        help="Quantization type (none, int8, int4).",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )

    args = parser.parse_args()
    main(
        args.is_vlm,
        args.model_name,
        args.prompt,
        args.image,
        args.compile,
        args.quantization,
        args.nb_samples,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
    )
