#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
import os
import signal
from typing import Any, Dict, Optional

import requests


def run_tgi_server(
    model_id: str,
    port: int = 8080,
    max_input_length: int = 4096,
    max_total_tokens: int = 8192,
    quantize: Optional[str] = None,
) -> subprocess.Popen:
    """Start the TGI server directly without Docker"""
    cmd = [
        "text-generation-launcher",
        "--model-id", model_id,
        "--port", str(port),
        "--max-input-length", str(max_input_length),
        "--max-total-tokens", str(max_total_tokens),
    ]
    
    if quantize:
        cmd.extend(["--quantize", quantize])
    
    print(f"Starting TGI server with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process


def test_server(port: int = 8080, max_retries: int = 20) -> Optional[Dict[str, Any]]:
    """Test if the server is running by sending a simple request"""
    url = f"http://localhost:{port}/generate"
    headers = {"Content-Type": "application/json"}
    data = {"inputs": "Hello, how are you?", "parameters": {"max_new_tokens": 20, "temperature": 0.7}}

    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server responded with status code {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"Server not ready yet, retrying in 5 seconds (attempt {i + 1}/{max_retries})")
            time.sleep(5)

    print("Failed to connect to the server after multiple attempts")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a model using text-generation-inference")
    parser.add_argument(
        "--model", type=str, default="jan-hq/Qwen3-14B-v0.1-deepresearch-100-step", help="Model ID from Hugging Face"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to serve the model on")
    parser.add_argument("--max-input-length", type=int, default=4096, help="Maximum input length")
    parser.add_argument("--max-total-tokens", type=int, default=8192, help="Maximum total tokens")
    parser.add_argument("--quantize", type=str, choices=["bitsandbytes", "bitsandbytes-nf4", "bitsandbytes-fp4", "gptq", "awq"], 
                        help="Quantization method to use (if any)")
    args = parser.parse_args()

    # Start the server
    process = run_tgi_server(
        model_id=args.model,
        port=args.port,
        max_input_length=args.max_input_length,
        max_total_tokens=args.max_total_tokens,
        quantize=args.quantize,
    )

    print(f"Server starting up, please wait...")
    time.sleep(30)  # Give the server some time to start

    # Test the server
    result = test_server(args.port)
    if result:
        print("Server is running! Sample response:")
        print(json.dumps(result, indent=2))
        print("\nKeep this terminal running to maintain the server.")
        print(f"The model is being served at: http://localhost:{args.port}")
        print("To stop the server, press Ctrl+C")

        try:
            # Wait for keyboard interrupt
            process.wait()
        except KeyboardInterrupt:
            print("Stopping server...")
            # Make sure to terminate all child processes
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    else:
        print("Failed to start the server properly")
        process.terminate() 