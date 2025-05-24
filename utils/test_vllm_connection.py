#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv
from smolagents import OpenAIServerModel

# Load environment variables
load_dotenv(override=True)

# Test vLLM connection
def test_vllm_connection():
    print("Testing vLLM server connection...")
    
    try:
        model = OpenAIServerModel(
            model_id="jan-hq/Qwen3-14B-v0.1-deepresearch-100-step",
            api_base="http://localhost:8000/v1/",
            api_key="EMPTY",
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        )
        
        print("✓ vLLM model initialized successfully")
        
        # Test a simple query
        test_messages = [{"role": "user", "content": "Hello, can you respond with just 'Hi there!'?"}]
        
        print("Testing model response...")
        response = model(test_messages, max_tokens=10, temperature=0.7)
        
        print(f"✓ Model response: {response}")
        print("✓ vLLM connection test successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ vLLM connection test failed: {str(e)}")
        print("Make sure vLLM server is running at http://localhost:8000")
        print("Start it with: vllm serve jan-hq/Qwen3-14B-v0.1-deepresearch-100-step --host 0.0.0.0 --port 8000")
        return False

if __name__ == "__main__":
    success = test_vllm_connection()
    sys.exit(0 if success else 1) 