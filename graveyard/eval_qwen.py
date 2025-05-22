#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name_or_path):
    """Load a Qwen model and tokenizer"""
    print(f"Loading model from: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    """Generate a response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, temperature=temperature, do_sample=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt) :].strip()

    return response


def evaluate_model(model, tokenizer, examples):
    """Evaluate the model on a list of examples"""
    results = []

    for i, (prompt, _) in enumerate(examples):
        print(f"\nEvaluating example {i + 1}/{len(examples)}...")
        response = generate_response(model, tokenizer, prompt)
        results.append({"prompt": prompt, "response": response})

    return results


def main():
    # Define example prompts to test the model
    examples = [
        ("What is machine learning?", ""),
        ("Explain the concept of reinforcement learning.", ""),
        ("Write a short poem about AI.", ""),
        ("What are the ethical concerns with large language models?", ""),
        ("Translate 'Hello, how are you?' to French.", ""),
    ]

    # Default model - smallest Qwen model (placeholder)
    default_model = "Qwen/Qwen1.5-0.5B"

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Qwen model")
    parser.add_argument("--model", type=str, default=default_model, help="Model name or path (default: %(default)s)")
    parser.add_argument(
        "--output", type=str, default="eval_results.csv", help="Output file path (default: %(default)s)"
    )
    args = parser.parse_args()

    # Load the model
    model, tokenizer = load_model(args.model)

    # Run evaluation
    print(f"\nRunning evaluation on {len(examples)} examples...")
    results = evaluate_model(model, tokenizer, examples)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
