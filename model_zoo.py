from trycast import isassignable
import torch
from typing import Union
from llm.model import LLMBase
from llm.chat import ChatFormat, Dialog


class Llama3(LLMBase):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3-8B-Instruct",
        compile: bool = True,
        quant: str = "int8",
    ):
        super().__init__(model_name, compile, quant)
        assert "Llama-3" in model_name, "This class is only for Llama-3 models"

    def apply_chat_template(
        self, conversation: Dialog, return_tensor: bool = True, **kwargs
    ) -> Union[str, torch.Tensor]:
        assert isassignable(
            conversation, Dialog
        ), "conversation must be a Dialog for Llama-3 model"
        assert isinstance(
            self.chat_template, ChatFormat
        ), "chat_template must be a ChatFormat for Llama-3 model"
        return self.chat_template.encode_dialog_prompt(
            conversation, return_tensor=return_tensor
        )

    @property
    def chat_template(self):
        return ChatFormat(self.tokenizer)