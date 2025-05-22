# Serving Qwen3-14B with Text Generation Inference

This README explains how to serve the jan-hq/Qwen3-14B-v0.1-deepresearch-100-step model using Hugging Face's Text Generation Inference (TGI).

## Prerequisites

- Docker with GPU support (nvidia-docker)
- Python 3.11+
- Required Python packages: requests, argparse

## Instructions

### 1. Start the TGI Server

Run the following command to start the TGI server with the Qwen3 model:

```bash
python serve_model.py
```

This will:
- Pull the TGI Docker image (if not already available)
- Start the server with the model jan-hq/Qwen3-14B-v0.1-deepresearch-100-step
- Serve the model on port 8080

Additional options:
```
--model           Model ID from Hugging Face (default: jan-hq/Qwen3-14B-v0.1-deepresearch-100-step)
--port            Port to serve the model on (default: 8080)
--gpu-id          GPU ID to use (default: 0)
--max-input-length  Maximum input length (default: 4096)
--max-total-tokens  Maximum total tokens (default: 8192)
```

### 2. Query the Model

Once the server is running, you can query the model using the provided script:

```bash
python query_model.py "Your prompt here"
```

Additional options:
```
--port          Port the model is being served on (default: 8080)
--max-tokens    Maximum number of new tokens to generate (default: 512)
--temperature   Sampling temperature (default: 0.7)
--top-p         Top-p sampling value (default: 0.9)
```

### Examples

Start the server with default settings:
```bash
python serve_model.py
```

Start with custom settings:
```bash
python serve_model.py --port 8090 --gpu-id 1 --max-input-length 2048 --max-total-tokens 4096
```

Query the model:
```bash
python query_model.py "Explain quantum computing in simple terms"
```

With custom parameters:
```bash
python query_model.py "Write a short story about a robot" --max-tokens 1024 --temperature 0.9
```

## Troubleshooting

1. If you see Docker permission errors, you may need to run with sudo or add your user to the docker group.

2. If the model fails to load, check that you have enough GPU memory. The Qwen3-14B model requires approximately 28GB of VRAM.

3. If connections to the server time out, try increasing the wait time in the serve_model.py script.

4. For connection issues, verify that the port is accessible and not blocked by a firewall. 