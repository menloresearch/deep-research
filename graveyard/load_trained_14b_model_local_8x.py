#!/usr/bin/env python3

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str = "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step"):
    # Verify available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")

    # Set environment variables for tensor parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configure for 8x GPUs with bfloat16 precision
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "balanced",  # Balance model across all available devices
        "max_memory": {i: "46GiB" for i in range(8)},  # Reserve 46GB per GPU (of 48GB)
        "low_cpu_mem_usage": True,  # Minimize CPU memory usage
    }

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model {model_id} across 8 GPUs...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    print("Model loaded successfully")
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
    print(f"Generating text with prompt: '{prompt[:50]}...' if longer")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Example usage
    prompt = "Explain the concept of deep learning in simple terms:"
    response = generate_text(model, tokenizer, prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
