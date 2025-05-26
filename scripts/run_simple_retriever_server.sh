#!/bin/bash

# Script to run the Simple Retriever server with customizable parameters.

# Default values
DEFAULT_SIMPLE_HOST="0.0.0.0"
DEFAULT_SIMPLE_PORT="8002"

# Use environment variables if set, otherwise use defaults
HOST_TO_USE="${SIMPLE_HOST:-$DEFAULT_SIMPLE_HOST}"
PORT_TO_USE="${SIMPLE_PORT:-$DEFAULT_SIMPLE_PORT}"

echo "Starting Simple Retriever server with the following parameters:"
echo "Host: ${HOST_TO_USE}"
echo "Port: ${PORT_TO_USE}"

CMD="uvicorn deploy.serving.serve_simple_retriever:app \
    --host \"${HOST_TO_USE}\" \
    --port ${PORT_TO_USE}"

exec ${CMD} 