# Serving Qwen3-14B with Text Generation Inference (Local Install)

This README explains how to serve the jan-hq/Qwen3-14B-v0.1-deepresearch-100-step model using Hugging Face's Text Generation Inference (TGI) without Docker.

## Prerequisites

- Python 3.11+
- CUDA-enabled GPU (for optimal performance)
- Required Python packages: requests, argparse

## Installation

1. Install the text-generation-inference package:

```bash
chmod +x install_tgi.sh
./install_tgi.sh
```

Or manually install it with:

```bash
pip install text-generation-inference
```

## Instructions

### 1. Start the TGI Server

Run the following command to start the TGI server with the Qwen3 model:

```bash
python serve_model_local.py
```

This will:
- Download the model (if not already cached)
- Start the server with the model jan-hq/Qwen3-14B-v0.1-deepresearch-100-step
- Serve the model on port 8080

Additional options:
```
--model           Model ID from Hugging Face (default: jan-hq/Qwen3-14B-v0.1-deepresearch-100-step)
--port            Port to serve the model on (default: 8080)
--max-input-length  Maximum input length (default: 4096)
--max-total-tokens  Maximum total tokens (default: 8192)
--quantize        Quantization method (choices: bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4, gptq, awq)
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
python serve_model_local.py
```

Start with 4-bit quantization (reduces VRAM requirements):
```bash
python serve_model_local.py --quantize bitsandbytes-nf4
```

Start with custom settings:
```bash
python serve_model_local.py --port 8090 --max-input-length 2048 --max-total-tokens 4096
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

1. If you encounter CUDA errors, make sure your CUDA drivers are up to date.

2. If the model fails to load, check that you have enough GPU memory. The Qwen3-14B model requires approximately 28GB of VRAM (less with quantization).

3. If you have limited VRAM, try using the `--quantize` option with `bitsandbytes-nf4` to run in 4-bit precision.

4. If connections to the server time out, try increasing the wait time in the serve_model_local.py script. 