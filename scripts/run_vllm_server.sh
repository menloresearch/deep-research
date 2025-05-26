#!/bin/bash

# Script to run the vLLM server with customizable parameters via environment variables.

# Default values
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3"
DEFAULT_VLLM_MODEL="Qwen/Qwen3-14B"
DEFAULT_VLLM_TP_SIZE="4"
DEFAULT_MAX_MODEL_LEN="8192"
DEFAULT_DTYPE="bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION="0.9"
DEFAULT_ENABLE_PREFIX_CACHING="True"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"

# Use environment variables if set, otherwise use defaults
CUDA_DEVICES_TO_USE="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
MODEL_TO_USE="${VLLM_MODEL:-$DEFAULT_VLLM_MODEL}"
TP_SIZE_TO_USE="${VLLM_TP_SIZE:-$DEFAULT_VLLM_TP_SIZE}"
MAX_MODEL_LEN_TO_USE="${VLLM_MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
DTYPE_TO_USE="${VLLM_DTYPE:-$DEFAULT_DTYPE}"
GPU_MEM_UTIL_TO_USE="${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
PREFIX_CACHING_TO_USE="${VLLM_ENABLE_PREFIX_CACHING:-$DEFAULT_ENABLE_PREFIX_CACHING}"
HOST_TO_USE="${VLLM_HOST:-$DEFAULT_HOST}"
PORT_TO_USE="${VLLM_PORT:-$DEFAULT_PORT}"

echo "Starting vLLM server with the following parameters:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES_TO_USE}"
echo "Model: ${MODEL_TO_USE}"
echo "Tensor Parallel Size: ${TP_SIZE_TO_USE}"
echo "Max Model Length: ${MAX_MODEL_LEN_TO_USE}"
echo "Dtype: ${DTYPE_TO_USE}"
echo "GPU Memory Utilization: ${GPU_MEM_UTIL_TO_USE}"
echo "Enable Prefix Caching: ${PREFIX_CACHING_TO_USE}"
echo "Host: ${HOST_TO_USE}"
echo "Port: ${PORT_TO_USE}"

# Set CUDA_VISIBLE_DEVICES for the python script
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES_TO_USE}

# Construct the command
CMD="python verifiers-deepresearch/verifiers/inference/vllm_serve.py \
    --model ${MODEL_TO_USE} \
    --tensor_parallel_size ${TP_SIZE_TO_USE} \
    --max_model_len ${MAX_MODEL_LEN_TO_USE} \
    --dtype ${DTYPE_TO_USE} \
    --gpu_memory_utilization ${GPU_MEM_UTIL_TO_USE} \
    --enable_prefix_caching ${PREFIX_CACHING_TO_USE} \
    --host \"${HOST_TO_USE}\" \
    --port ${PORT_TO_USE}"

# Execute the command
# Using exec to replace the shell process with the python process,
# which can be helpful for signal handling (e.g., when make tries to kill it).
exec ${CMD} 