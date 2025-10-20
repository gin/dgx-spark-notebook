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
Serve with vLLM
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
TBA
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
