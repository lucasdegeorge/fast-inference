import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
import torchvision.transforms as transforms
from typing import Union


class ViTImageEncoderHF:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, revision="bfloat16"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, do_resize=True, resample=True, do_rescale=True, do_normalize=True
        )
        self.vision_model = self.model.vision_tower
        self.projector = self.model.multi_modal_projector
        self.max_new_tokens = 100

    def infer(self, prompt: str, image: str) -> str:
        raw_image = Image.open(image)
        inputs = self.processor(prompt, raw_image, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    # TODO: check if processor is mandatory and if it is a time bottleneck
    def get_image_embeddings(
        self, image: Union[str, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(image, str):
            raw_image = Image.open(image)
        elif isinstance(image, Image.Image):
            raw_image = image
        elif isinstance(image, torch.Tensor):
            ## TODO: Add further checks on the tensor shape
            raw_image = transforms.ToPILImage()(image)
        else:
            raise ValueError("Image must be a str; a torch.Tensor or a PIL image")
        inputs = self.processor(
            "", raw_image, return_tensors="pt"
        )  ## TODO: Check if necessary
        vision_encoded = self.vision_model(inputs.pixel_values.to(dtype=torch.bfloat16))
        outputs = self.projector(vision_encoded.last_hidden_state)
        return outputs
