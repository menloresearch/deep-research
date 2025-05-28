#!/bin/bash

# Serve Jan HQ Qwen3-14B model with reasoning enabled
vllm serve jan-hq/Qwen3-14B-v0.2-deepresearch-300-step \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --port 8000 \
    --tensor-parallel-size 8