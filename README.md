# fast_inference

PaliGemma fast inference - LLM fast inference

Efficient pytorch-native transformer text generation based on [gpt-fast](https://github.com/pytorch-labs/gpt-fast)

## Models supported/tested

### VLMs

```text
paligemma-3b-mix-224
```

### LLMs

```text
Llama-2-7b-chat-hf
Llama-2-13b-chat-hf
Meta-Llama-3-8B
Meta-Llama-3-8B-Instructs
Mistral-7B-Instruct-v0.2
TinyLlama-1.1B-Chat-v1.0
TinyLlama-1.1B-intermediate-step-1431k-3T
gemma-7b-it
gemma-2b-it
```

## Example: Benchmark tokens/s

### For a VLM

```bash
python main.py --is_vlm --compile --model_name paligemma-3b-mix-224 --prompt "Describe this image" --image /path/to/image.jpg --quantization int8
```

or

```bash
bash benchmarck_vlm.sh paligemma-3b-mix-224 /path/to/image.jpg
```

### For an LLM

```bash
python main.py --compile --model_name Meta-Llama-3-8B --prompt "Hello, my name is" --quantization int8
```

or

```bash
bash benchmarck_llm.sh Meta-Llama-3-8b
```

## Example: Inference

### PaliGemma inference

```python
from vlm.model import GemmaVLMBase

model = GemmaVLMBase(model_name="paligemma-3b-mix-224", compile=True, quant="int8")

prompt = "Give me a very descriptive caption of this image.\n"
image = "tennis_player.jpeg"
print(model.generate(prompt, image, 200, temperature=0.8, top_k=200))
```


### LLM Chat inference with Llama-3 Instruct

```python
from model_zoo import Llama3

model = Llama3(model_name="Meta-Llama-3-8B-Instruct", quant="int8")

dialog = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I need help with my computer."},
]

updated_dialog = model.generate(prompt, 200, clean_dialog=True, temperature=0.8, top_k=200)

## Print the answer of the model
print(updated_dialog[-1]["content"])
```


### LLM inference

```python
from llm.model import LLMBase

model = LLMBase(model_name="Meta-LLama-3-8B", compile=True, quant="int8")

prompt = "Hello, my name is \n"
print(model.generate(prompt, 200, temperature=0.8, top_k=200))
```

or for Gemma-based model

```python
from llm.model import GemmaBase

model = GemmaBase(model_name="gemma-7b-it", compile=True, quant="int8")

prompt = "Hello, my name is \n"
print(model.generate(prompt, 200, temperature=0.8, top_k=200))
```



## How to donwload models?

### PaliGemma

```bash
bash ./download_hf/prepare_paligemma.sh google/paligemma-3b-mix-224
```
One can replace 'paligemma-3b-mix-224' by any (PyTorch) checpoint from the [PaliGemma collection](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda)


### LLM

```bash
bash ./download_hf/prepare_paligemma.sh $MODEL_REPO
```


## TO DO

- PaliGemma Vision encoder implementation without HF (in progress)
- Add new VLM models
- Add chat-template support for SentencePiece-based LLMs and VLMs
- Add multi-sample generation
- Batched inference for LLM and VLM
