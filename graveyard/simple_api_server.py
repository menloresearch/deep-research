#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from threading import Thread
from typing import Any, Dict, List, Literal, Optional, Union, cast

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers.generation.streamers import TextIteratorStreamer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

app = FastAPI(title="Simple OpenAI-compatible API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
TOKENIZER = None


# Pydantic models for API
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]


def load_model(model_name: str, device: str, quantize: bool = False) -> None:
    global MODEL, TOKENIZER

    print(f"Loading model {model_name}...")
    start_time = time.time()

    # Load tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    if quantize:
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True, trust_remote_code=True
        )
    else:
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")


def format_chat_prompt(messages: List[Message]) -> str:
    """Format messages into a chat prompt for the model"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            # Format system message (customize based on model requirements)
            prompt += f"<|system|>\n{msg.content}\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content}\n"
    # Add the final assistant prompt
    prompt += "<|assistant|>\n"
    return prompt


def generate_response(
    messages: List[Message], max_tokens: int = 256, temperature: float = 0.7, top_p: float = 1.0
) -> str:
    if MODEL is None or TOKENIZER is None:
        raise ValueError("Model or tokenizer not loaded")

    prompt = format_chat_prompt(messages)

    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    # Generate
    with torch.no_grad():
        outputs = MODEL.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
        )

    # Decode and extract only the new content
    full_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt) :]

    return generated_text.strip()


async def generate_stream(
    messages: List[Message], max_tokens: int = 256, temperature: float = 0.7, top_p: float = 1.0
):
    if MODEL is None or TOKENIZER is None:
        raise ValueError("Model or tokenizer not loaded")

    prompt = format_chat_prompt(messages)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    streamer = TextIteratorStreamer(TOKENIZER, skip_special_tokens=True)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "streamer": streamer,
    }

    # Start generation in a separate thread
    thread = Thread(target=lambda: MODEL.generate(**generation_kwargs) if MODEL is not None else None)
    thread.start()

    # Stream the response
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        response_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL.config._name_or_path if MODEL is not None else "unknown",
            "choices": [{"index": 0, "delta": {"content": new_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(response_data)}\n\n"
        await asyncio.sleep(0)

    # Send the final chunk with finish_reason
    final_data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL.config._name_or_path if MODEL is not None else "unknown",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Handle optional parameters with defaults
    max_tokens = request.max_tokens if request.max_tokens is not None else 256
    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 1.0

    if request.stream:
        return StreamingResponse(
            generate_stream(request.messages, max_tokens, temperature, top_p), media_type="text/event-stream"
        )
    else:
        response_text = generate_response(request.messages, max_tokens, temperature, top_p)

        input_tokens = len(TOKENIZER.encode(format_chat_prompt(request.messages)))
        output_tokens = len(TOKENIZER.encode(response_text))

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=MODEL.config._name_or_path,
            choices=[Choice(index=0, message=Message(role="assistant", content=response_text), finish_reason="stop")],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL.config._name_or_path if MODEL else "model-not-loaded",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Start a simple OpenAI-compatible API server")
    parser.add_argument(
        "--model", type=str, default="jan-hq/Qwen3-14B-v0.1-deepresearch-100-step", help="Model ID to load"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to load model on (auto, cuda:0, etc.)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")
    args = parser.parse_args()

    load_model(args.model, args.device, args.quantize)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
