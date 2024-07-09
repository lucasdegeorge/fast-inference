import torch
from pathlib import Path
import time
from typing import Tuple, Union
import contextlib
from PIL import Image
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vlm.image_encoder_hf import ViTImageEncoderHF
from vlm.text_encoder.base import TransformerForVLM
from vlm.text_encoder.gemma import TransformerForGemmaVLM
from vlm.utils import sample_vlm as sample
from llm.tokenizer import get_tokenizer

# import torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs = True


class VLMBase:
    def __init__(
        self, model_name: str, compile: bool = True, quant: str = "none"
    ) -> None:
        """
        'model_name' dict with key 'text' and 'image'
        'quant' is right know (with hf models) only used for the text encoder
        """
        self.model_name = model_name
        self.compile = compile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        ## Text encoder
        self.text_model = self.load_text_model(quant)
        tokenizer_path = self.text_checkpoint_path / "tokenizer.model"
        assert tokenizer_path.is_file(), str(tokenizer_path)
        self.tokenizer = get_tokenizer(tokenizer_path, model_name)
        self.eos_id = self.tokenizer.eos_id()

        # TODO: Implement support for other non-HF image encoders
        self.image_encoder = ViTImageEncoderHF(self.text_model.config.hf_checkpoint)

    def load_text_model(self, quant: str = "none"):
        """Similar to 'load_model' from LLMBase."""
        tick = time.time()
        with torch.device("meta"):
            model = TransformerForVLM.from_name(self.model_name)
            self.text_checkpoint_path = model.config.llm_args.checkpoint_path
            if isinstance(self.text_checkpoint_path, str):
                self.text_checkpoint_path = Path(self.text_checkpoint_path)

        if quant == "none":
            self.model_path = self.text_checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.text_checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.text_checkpoint_path / "model_int4.g32.pth"
            assert self.model_path.is_file(), str(self.model_path)
            path_comps = self.model_path.name.split(".")
            groupsize = int(path_comps[-2][1:])
            from llm.quantization import WeightOnlyInt4QuantHandler

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        else:
            raise ValueError(f"Invalid quantization type: {quant}")

        checkpoint = torch.load(str(self.model_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

        model = model.to(
            device=self.device, dtype=torch.bfloat16
        )  ## TO DO: check add support for other dtypes
        print(f"Model loaded in {time.time() - tick:.02f} seconds")
        return model.eval()

    def tokenize(self, string: str, bos: bool = True) -> torch.Tensor:
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        return torch.tensor(tokens, dtype=torch.int, device=self.device)

    def decode_one_token(
        self, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = self.text_model(x, input_pos=input_pos)
        return sample(logits, **sampling_kwargs)

    def decode_n_tokens(  ## TODO: check for call-back !
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        stop_first_eos: bool = True,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                next_token, next_prob = self.decode_one_token(
                    cur_token, input_pos, **sampling_kwargs
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                new_probs.append(next_prob.clone())
                cur_token = next_token.view(1, -1)
                if stop_first_eos and next_token == self.eos_id:
                    # print(f"Found EOS token at position {i}")
                    break
        return new_tokens, new_probs

    def prefill(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor,
        input_pos: torch.Tensor,
        **sampling_kwargs,
    ) -> torch.Tensor:
        # input_pos: [B, S]
        logits = self.text_model(x, input_pos=input_pos, embeds=embeds)
        return sample(logits, **sampling_kwargs)[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[torch.Tensor, str],
        image: Union[str, Image.Image, torch.Tensor],
        max_new_tokens: int,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        remove_image_tokens: bool = True,
        **sampling_kwargs,
    ) -> torch.Tensor:

        # Preprocessing
        prof = contextlib.nullcontext()

        if self.compile:  # TO DO: check and if add compile_prefill
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            # self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        if do_decode and not remove_image_tokens:
            raise ValueError("Cannot return decoded sequence with image tokens")

        # Prompts and Embeddings
        if isinstance(prompt, str):
            prompt = self.tokenize(prompt, bos=True)

        text_embeddings = self.text_model.tok_embeddings(prompt)
        image_embeddings = self.image_encoder.get_image_embeddings(image)
        image_embeddings /= (
            2048**0.5
        )  # TODO: check if this is necessary and replace by config value
        ti_embeddings = torch.cat(
            [image_embeddings[0], text_embeddings], dim=0
        ).unsqueeze(0)

        image_prompt = torch.full(
            (self.text_model.config.num_image_tokens,),
            self.text_model.config.image_token_index,
            device=prompt.device,
            dtype=prompt.dtype,
        )
        prompt = torch.cat([image_prompt, prompt], dim=0)

        # Generation
        with prof:
            T = prompt.size(0)
            T_max = T + max_new_tokens
            max_seq_length = min(T_max, self.text_model.config.llm_args.block_size)

            self.text_model.setup_caches(
                max_batch_size=1, max_seq_length=max_seq_length, device=prompt.device
            )
            input_pos = torch.arange(0, T, device=prompt.device)
            next_token = self.prefill(
                prompt.view(1, -1),
                input_pos=input_pos,
                embeds=ti_embeddings,
                **sampling_kwargs,
            ).clone()

            # Generate
            input_pos = torch.tensor([T], device=prompt.device, dtype=torch.int)
            generated_tokens, _ = self.decode_n_tokens(
                next_token.view(1, -1),
                input_pos,
                max_new_tokens - 1,
                stop_first_eos,
                **sampling_kwargs,
            )

            # Fill an empty tensor with the generated tokens
            T_new = T + len(generated_tokens) + 1
            empty = torch.empty(T_new, dtype=prompt.dtype, device=prompt.device)
            empty[:T] = prompt
            seq = empty
            seq[T] = next_token
            seq[T + 1 :] = torch.cat(generated_tokens)

        if only_new_tokens:
            seq = seq[T:]
        elif remove_image_tokens:
            seq = seq[self.text_model.config.num_image_tokens :]

        if do_decode:
            seq = self.tokenizer.decode(seq.tolist())
            return seq
        else:
            return seq

    def benchmark_tok_per_s(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        max_new_tokens: int,
        nb_samples: int = 100,
        **sampling_kwargs,
    ) -> float:
        tokens_per_s = list()
        for i in range(-1, nb_samples):
            t0 = time.perf_counter()
            encoded = self.tokenize(prompt, bos=True)
            output = self.generate(
                encoded,
                image,
                max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                remove_image_tokens=True,
                **sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                tokens_generated = output.size(0) - encoded.size(0)
                tokens_per_s.append(tokens_generated / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        return np.mean(tokens_per_s)


class GemmaVLMBase(VLMBase):
    def __init__(
        self, model_name: str, compile: bool = True, quant: str = "none"
    ) -> None:
        super().__init__(model_name, compile, quant)

    def load_text_model(self, quant: str = "none"):
        """Similar to 'load_model' from GemmaBase."""
        tick = time.time()
        with torch.device("meta"):
            model = TransformerForGemmaVLM.from_name(self.model_name)
            self.text_checkpoint_path = model.config.llm_args.checkpoint_path
            if isinstance(self.text_checkpoint_path, str):
                self.text_checkpoint_path = Path(self.text_checkpoint_path)

        if quant == "none":
            self.model_path = self.text_checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.text_checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.text_checkpoint_path / "model_int4.g32.pth"
            assert self.model_path.is_file(), str(self.model_path)
            path_comps = self.model_path.name.split(".")
            groupsize = int(path_comps[-2][1:])
            from llm.quantization import WeightOnlyInt4QuantHandler

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        else:
            raise ValueError(f"Invalid quantization type: {quant}")

        checkpoint = torch.load(str(self.model_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

        model = model.to(
            device=self.device, dtype=torch.bfloat16
        )  ## TO DO: check add support for other dtypes
        print(f"Model loaded in {time.time() - tick:.02f} seconds")
        return model.eval()
