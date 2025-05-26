#!/bin/bash

# Script to run the Sandbox server with customizable parameters.

# Default values
DEFAULT_SANDBOX_HOST="0.0.0.0"
DEFAULT_SANDBOX_PORT="8005"

# Use environment variables if set, otherwise use defaults
HOST_TO_USE="${SANDBOX_HOST:-$DEFAULT_SANDBOX_HOST}"
PORT_TO_USE="${SANDBOX_PORT:-$DEFAULT_SANDBOX_PORT}"

echo "Starting Sandbox server with the following parameters:"
echo "Host: ${HOST_TO_USE}"
echo "Port: ${PORT_TO_USE}"

CMD="uvicorn deploy.serving.serve_sandbox_env:app \
    --host \"${HOST_TO_USE}\" \
    --port ${PORT_TO_USE}"

exec ${CMD} 