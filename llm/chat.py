from typing import TypedDict, Sequence, Literal, List, Union
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.tokenizer import TiktokenWrapper


### Adapted from https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class ChatFormat:
    """Format a dialog into a chat prompt for a Tiktoken-based model."""

    def __init__(self, tokenizer: TiktokenWrapper):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(
        self, dialog: Dialog, return_tensor: bool = True
    ) -> Union[List[int], torch.Tensor]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        if return_tensor:
            return torch.tensor(tokens, dtype=torch.int32, device="cuda")
        return tokens
