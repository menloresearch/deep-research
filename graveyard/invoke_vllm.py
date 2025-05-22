#!/usr/bin/env python3
import json
import sys
from typing import Optional

from openai import OpenAI


def chat_with_model(
    prompt: str,
    api_key: str = "EMPTY",
    base_url: str = "http://localhost:8000/v1",
    model: Optional[str] = "jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    try:
        # Initialize OpenAI client with vLLM settings
        client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            # Fetch available models
            models = client.models.list()
            available_models = [m.id for m in models.data]
            print("Available models:", available_models)

            # Validate the model exists
            if model and model not in available_models:
                print(f"Warning: Specified model {model} not found. Using first available model.")
                model = available_models[0] if available_models else None

        except Exception as model_list_error:
            print(f"Error listing models: {model_list_error}")
            raise

        if not model:
            raise ValueError("No models available on the server")

        print(f"Using model: {model}")

        # Prepare the chat completion request
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Print full response for debugging
            print("Full response:", json.dumps(response.model_dump(), indent=2))

            # Return the model's response
            generated_text = response.choices[0].message.content
            print("Generated text:", generated_text)
            return generated_text or "No response generated."

        except Exception as completion_error:
            print(f"Error during chat completion: {completion_error}")
            raise

    except Exception as e:
        print(f"Unexpected error communicating with the model: {e}")
        sys.exit(1)


def main():
    # Allow custom prompt via command line argument
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain quantum computing in simple terms"

    response = chat_with_model(prompt)
    print(f"\nFinal Model's response:\n{response}")


if __name__ == "__main__":
    main()
