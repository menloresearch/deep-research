#!/usr/bin/env python3

import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def setup_hf_api(api_token: Optional[str] = None) -> dict:
    if api_token is None:
        api_token = os.environ.get("HF_TOKEN")
        if api_token is None:
            raise ValueError("No API token provided. Set HF_TOKEN environment variable or pass token to function.")

    headers = {"Authorization": f"Bearer {api_token}"}
    return headers


def generate_text(model_id: str, prompt: str, headers: dict, max_length: int = 512, temperature: float = 0.7) -> str:
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": max_length, "temperature": temperature, "do_sample": True},
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    result = response.json()

    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "generated_text" in result[0]:
            return result[0]["generated_text"]

    # Fallback to returning the raw result if format is unexpected
    return str(result)


if __name__ == "__main__":
    # Setup API with your token
    model_id = "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step"
    headers = setup_hf_api()  # Set HF_API_TOKEN env var or pass your token here

    # Example usage
    prompt = "Explain the concept of deep learning in simple terms:"
    response = generate_text(model_id, prompt, headers)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
