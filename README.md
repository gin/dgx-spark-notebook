# DGX Spark Notebook
Notes and things I noticed while setting up a DGX Spark mini-PC.


## To Do's after booting up
- [Setup remote login](#setup-remote-login)
- [Setup chatbot](#setup-chatbot)
- [Setup code copilot](#setup-code-copilot)
- [Fine-tune an existing model](#fine-tune-an-existing-model)
    - [Test the fine-tuned model](#test-the-fine-tuned-model)
    - [Progressive fine-tuning](#progressive-fine-tuning)
- [Optimizations](#optimizations)
    - [Thinking](#thinking)
    - [Speculative decoding](#speculative-decoding)
    - [NIM](#NIM)
- [etc](#etc)

----
### Setup remote login
```
Disable password login fallback
```


----
### Setup chatbot
```
Open WebUI with models from Ollama
$ docker pull ghcr.io/open-webui/open-webui:ollama

$ docker run -d -p 8080:8080 --gpus=all \
    -v open-webui:/app/backend/data \
    -v open-webui-ollama:/root/.ollama \
    --name open-webui \
    ghcr.io/open-webui/open-webui:ollama
```

Pick the smartest model that can fit in 128GB memory.  
Per 2025-10, 4-bit quantized is fine.  
Any lower will increasingly degrade in perplexity.  

- https://lmarena.ai/leaderboard
- https://artificialanalysis.ai/models


```
Benchmark

model           size        context     response    prompt      in-mem-size
---------------------------------------------------------------------------
gpt-oss:20b     14GB        128k        50tok/s     1800tok/s   30GB
qwen3-coder:30b 19GB        256k        60tok/s      850tok/s   70GB
```


----
### Setup code copilot
```
VSCode extension: Continue.dev 

To use the already downloaded models from open-webui:ollama, expose port 11434 and set OLLAMA_HOST.

$ docker run -d \
  -p 8080:8080 \
  -p 11434:11434 \
  --gpus=all \
  -e OLLAMA_HOST=0.0.0.0 \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui \
  ghcr.io/open-webui/open-webui:ollama
```

```
Continue.dev's model setting, add:

    {
      "title": "Qwen 3 Coder",
      "provider": "ollama",
      "model": "qwen3-coder:30b",
      "apiBase": "http://<IP_ADDRESS>:11434",
      "systemMessage": "You are an expert software developer. You give helpful and concise responses. Whenever you write a code block you include the language after the opening ticks."
    }
```


----
### Fine-tune an existing model
```
Unsloth
Pytouch
NeMo (Nvidia)
Dreambooth (images)
```

```
Unsloth

dep:
    nvidia-cuda-toolkit
    pytorch

    transformers
    peft
    datasets

    unsloth
    unsloth_zoo

    bitsandbytes
```

```
PyTorch

dep:
    transformers
    peft
    datasets
    trl
    bitsandbytes
```

e.g.
```python
import pytorch
from trl import SFTConfig, SFTTrainer
# Config, load dataset, then:
trainer = SFTTrainer(...)
trainer_stats = trainer.train()
```

Output:
https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/Llama3_8B_LoRA_finetuning.py
```
============================================================
TRAINING COMPLETED
============================================================
Training runtime: 76.69 seconds
Samples per second: 6.52
Steps per second: 1.63
Train loss: 1.0082
============================================================
```

```
Serve with vLLM
```

Huggingface.co gives free 5TB public, 100GB private model storage  
    specify <USERNAME_OR_ORG>/<MODEL_NAME>, <WRITE_ACCESS_TOKEN>  
    https://huggingface.co/settings/tokens/new?tokenType=write

e.g.
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(...)
# Some fine-tuning steps, then:
model.push_to_hub_merged("hLuigi/gpt_oss_20B_RL_2048_Game", tokenizer, token = "hf_123ABC", save_method = "mxfp4")
```

#### Test the fine-tuned model
```
TBA
```

#### Progressive fine-tuning
```
TBA
```


----
### Optimizations
```
fp16 is standard
    because GPUs older than hopper does not support fp8
    some report fp16 scales better than bp16

mxfp4
    hopper and newer architectures support it (e.g. RTX 50xx, Hx00, Bx00)
```

#### Thinking
```
Chain-of-thought
Draft–critique–revise-repeat
Planning
```

#### Speculative decoding
```
TBA
```

#### NIM
```
Container
"Nvidia Inference Microservice"
```

### etc
```
flush the buffer cache using:

sudo sh -c ‘sync; echo 3 > /proc/sys/vm/drop_caches’
```
