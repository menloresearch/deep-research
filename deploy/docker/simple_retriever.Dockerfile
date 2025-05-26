# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY deploy/requirements.txt .

# Install uv and then project dependencies
RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy the application code into the container
COPY deploy/serving/serve_simple_retriever.py ./deploy/serving/
COPY src/ ./src/

# Make port available to the world outside this container
EXPOSE 8002

# Define environment variables
ENV SIMPLE_RETRIEVER_PORT=8002
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app

# Run the application using the script's __main__ block
CMD ["python", "deploy/serving/serve_simple_retriever.py", "--host", "0.0.0.0", "--port", "${SIMPLE_RETRIEVER_PORT:-8002}"] 