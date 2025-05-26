#!/bin/bash

# Script to run the Mock Retriever server with customizable parameters.

# Default values
DEFAULT_MOCK_HOST="0.0.0.0"
DEFAULT_MOCK_PORT="8003"

# Use environment variables if set, otherwise use defaults
HOST_TO_USE="${MOCK_HOST:-$DEFAULT_MOCK_HOST}"
PORT_TO_USE="${MOCK_PORT:-$DEFAULT_MOCK_PORT}"

echo "Starting Mock Retriever server with the following parameters:"
echo "Host: ${HOST_TO_USE}"
echo "Port: ${PORT_TO_USE}"

CMD="uvicorn deploy.serving.serve_mock_retriever:app \
    --host \"${HOST_TO_USE}\" \
    --port ${PORT_TO_USE}"

exec ${CMD} 