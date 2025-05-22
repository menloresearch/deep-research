# Simple OpenAI-Compatible API Server

This is a lightweight server that loads the Qwen3 model using transformers and serves it with an OpenAI-compatible API.

```json
{
  "model": "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
  "messages": [
    {
      "role": "system",
      "content": "You are a research assistant with deep knowledge of scientific literature."
    },
    {
      "role": "user",
      "content": "What are the key differences between deep learning and traditional machine learning approaches?"
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

## Features

- OpenAI-compatible API endpoints
- Support for chat completions
- Support for streaming responses
- Built with FastAPI for high performance
- Quantization support for lower memory usage

## Prerequisites

- Python 3.11+
- CUDA-enabled GPU (recommended)
- Required Python packages (see `requirements.txt`)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the server

Run the following command to start the server with the default model:

```bash
python simple_api_server.py
```

This will:

- Load the jan-hq/Qwen3-14B-v0.1-deepresearch-100-step model
- Start a server on <http://0.0.0.0:8000>
- Expose OpenAI-compatible API endpoints

### Command-line options

```
--model        Model ID from Hugging Face (default: jan-hq/Qwen3-14B-v0.1-deepresearch-100-step)
--device       Device to load model on (auto, cuda:0, etc.) (default: auto)
--port         Port to run server on (default: 8000)
--host         Host to run server on (default: 0.0.0.0)
--quantize     Enable 4-bit quantization (reduces VRAM usage)
```

### Example with custom settings

```bash
python simple_api_server.py --model jan-hq/Qwen3-14B-v0.1-deepresearch-100-step --port 8080 --quantize
```

## API Usage

### Chat Completions

Endpoint: `/v1/chat/completions`

Example with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is deep learning?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Streaming

Add `"stream": true` to your request to get a streaming response:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short story about a robot."}
    ],
    "max_tokens": 250,
    "temperature": 0.8,
    "stream": true
  }'
```

### List Available Models

Endpoint: `/v1/models`

```bash
curl http://localhost:8000/v1/models
```

## Using with OpenAI-compatible clients

You can use this server with any OpenAI-compatible client by setting the base URL to your server's address:

### Python example

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used but required
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Troubleshooting

1. If you encounter out-of-memory errors, try using the `--quantize` flag to enable 4-bit quantization

2. If the model is loading slowly, make sure you have a fast internet connection for the initial download

3. For optimal performance, use a GPU with at least 16GB of VRAM (more for non-quantized models)
