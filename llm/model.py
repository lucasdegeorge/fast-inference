import torch
import time
from pathlib import Path
from typing import Tuple
import torch._dynamo.config
import torch._inductor.config
import numpy as np
from typing import Dict, List, Tuple, Union
import contextlib
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.architecture.base import Transformer
from llm.tokenizer import get_tokenizer
from llm.utils import sample
from llm.architecture.gemma import TransformerForGemma

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


class LLMBase:
    def __init__(
        self, model_name: str, compile: bool = True, quant: str = "none"
    ) -> None:
        self.model_name = model_name
        self.compile = compile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self.model = self.load_model(quant=quant)

        tokenizer_path = self.checkpoint_path / "tokenizer.model"
        assert tokenizer_path.is_file(), str(tokenizer_path)
        self.tokenizer = get_tokenizer(tokenizer_path, model_name)
        self.eos_id = self.tokenizer.eos_id()

        # For Jinja template: a dict mapping special tokens (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).
        # Special tokens can be added to this dict using the `add_special_token` method.
        self.special_tokens_map = {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
        }
        # self.special_tokens_map = {'bos_token': "<s>", 'eos_token': "</s>"}

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def load_model(self, quant: str = "none"):
        tick = time.time()
        with torch.device("meta"):
            model = Transformer.from_name(self.model_name)
            self.checkpoint_path = model.config.checkpoint_path
            if isinstance(self.checkpoint_path, str):
                self.checkpoint_path = Path(self.checkpoint_path)

        if quant == "none":
            self.model_path = self.checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.checkpoint_path / "model_int4.g32.pth"
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
        logits = self.model(x, input_pos)
        return sample(logits, **sampling_kwargs)

    def decode_n_tokens(
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
        self, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
    ) -> torch.Tensor:
        # input_pos: [B, S]
        logits = self.model(x, input_pos)
        return sample(logits, **sampling_kwargs)[0]

    ## TO DO: add multi-sample generation
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        **sampling_kwargs,
    ) -> torch.Tensor:

        # Preprocessing
        prof = contextlib.nullcontext()

        if self.compile:  # TO DO: check and if add compile_prefill
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            # self.prefill = torch.compile(self.prefill, fullgraph=True, dynamic=True)

        if isinstance(prompt, str):
            prompt = self.tokenize(prompt, bos=True)

        # Generation
        with prof:
            T = prompt.size(0)
            T_max = T + max_new_tokens
            max_seq_length = min(T_max, self.model.config.block_size)

            self.model.setup_caches(
                max_batch_size=1, max_seq_length=max_seq_length, device=prompt.device
            )
            input_pos = torch.arange(0, T, device=prompt.device)
            next_token = self.prefill(
                prompt.view(1, -1), input_pos, **sampling_kwargs
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

            if do_decode:
                seq = self.tokenizer.decode(seq.tolist())

            return seq

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        """self.chat_template must be defined before calling this method"""

        try:
            compiled_template = self._compile_jinja_template(self.chat_template)
        except NotImplementedError:
            raise NotImplementedError(
                "A chat template must be defined before calling this method."
            )

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple))
            or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        rendered = []
        template_kwargs = {**self.special_tokens_map, **kwargs}
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            rendered_chat = compiled_template.render(
                messages=chat,
                add_generation_prompt=add_generation_prompt,
                **self.special_tokens_map,
            )
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        return rendered

    @property
    def chat_template(self):
        """This method should be overridden by subclasses.
        It should return a Jinja template string that can be used to render a chat conversation.
        See the corresponding template in the Hugging Face API
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
        # Example template
        # template = (
        #     "{% if messages[0]['role'] == 'system' %}"
        #     "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        #     "{% set system_message = messages[0]['content'] %}"
        #     "{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}"  # Here USE_DEFAULT_PROMPT replaced by "false"
        #     "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
        #     "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        #     "{% else %}"
        #     "{% set loop_messages = messages %}"
        #     "{% set system_message = false %}"
        #     "{% endif %}"
        #     "{% for message in loop_messages %}"  # Loop over all non-system messages
        #     "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        #     "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        #     "{% endif %}"
        #     "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
        #     "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        #     "{% else %}"
        #     "{% set content = message['content'] %}"
        #     "{% endif %}"
        #     "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        #     "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
        #     "{% elif message['role'] == 'system' %}"
        #     "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
        #     "{% elif message['role'] == 'assistant' %}"
        #     "{{ ' '  + content.strip() + ' ' + eos_token }}"
        #     "{% endif %}"
        #     "{% endfor %}"
        # )
        # default_message = ""
        # template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        # return template

    def _compile_jinja_template(self, chat_template):
        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def add_special_token(self, tokens: dict):
        for key, value in tokens.items():
            self.special_tokens_map[key] = value

    def benchmark_tok_per_s(
        self, prompt: str, max_new_tokens: int, nb_samples: int = 5, **sampling_kwargs
    ) -> float:
        tokens_per_s = list()
        for i in range(-1, nb_samples):
            t0 = time.perf_counter()
            encoded = self.tokenize(prompt, bos=True)
            output = self.generate(
                encoded,
                max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                **sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                # print(self.tokenizer.decode(output.tolist()))
                # print("-----------------------")
                tokens_generated = output.size(0) - encoded.size(0)
                tokens_per_s.append(tokens_generated / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        return np.mean(tokens_per_s)


class GemmaBase(LLMBase):
    def __init__(
        self, model_name: str, compile: bool = True, quant: str = "none"
    ) -> None:
        super().__init__(model_name, compile, quant)

    def load_model(self, quant: str = "none"):
        tick = time.time()
        with torch.device("meta"):

            ## New line for Gemma models
            model = TransformerForGemma.from_name(self.model_name)

            self.checkpoint_path = model.config.checkpoint_path
            if isinstance(self.checkpoint_path, str):
                self.checkpoint_path = Path(self.checkpoint_path)

        if quant == "none":
            self.model_path = self.checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.checkpoint_path / "model_int4.g32.pth"
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
