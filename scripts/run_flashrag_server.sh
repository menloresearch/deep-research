#!/bin/bash

# Script to run the FlashRAG server with customizable parameters.

# Default values
DEFAULT_FLASHRAG_HOST="0.0.0.0"
DEFAULT_FLASHRAG_PORT="8001"
DEFAULT_FLASHRAG_CONFIG_PATH="deploy/serving/retriever_config.yaml"
DEFAULT_FLASHRAG_NUM_RETRIEVER="1"

# Use environment variables if set, otherwise use defaults
HOST_TO_USE="${FLASHRAG_HOST:-$DEFAULT_FLASHRAG_HOST}"
PORT_TO_USE="${FLASHRAG_PORT:-$DEFAULT_FLASHRAG_PORT}"
CONFIG_PATH_TO_USE="${FLASHRAG_CONFIG_PATH:-$DEFAULT_FLASHRAG_CONFIG_PATH}"
NUM_RETRIEVER_TO_USE="${FLASHRAG_NUM_RETRIEVER:-$DEFAULT_FLASHRAG_NUM_RETRIEVER}"

echo "Starting FlashRAG server with the following parameters:"
echo "Host: ${HOST_TO_USE}" # Note: deploy/serving/serve_flashrag_retriever.py internally uses 0.0.0.0 for uvicorn
echo "Port: ${PORT_TO_USE}"
echo "Config Path: ${CONFIG_PATH_TO_USE}"
echo "Number of Retrievers: ${NUM_RETRIEVER_TO_USE}"

CMD="python deploy/serving/serve_flashrag_retriever.py \
    --port ${PORT_TO_USE} \
    --config \"${CONFIG_PATH_TO_USE}\" \
    --num_retriever ${NUM_RETRIEVER_TO_USE}"

# The serve_flashrag_retriever.py script internally starts uvicorn with host 0.0.0.0
# If it were to take --host, we would add: --host \"${HOST_TO_USE}\" \

exec ${CMD} 