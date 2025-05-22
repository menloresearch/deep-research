#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str = "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step"):
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
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

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
