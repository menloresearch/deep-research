#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix --exclude third_party --exclude notebooks
	ruff format --exclude third_party --exclude notebooks


## Run tests
.PHONY: test
test:
	python -m pytest tests


## data
.PHONY: data
data:
	@echo "Downloading Musique dataset..."
	bash scripts/prepare_train_data/download_data_musique.sh
	@echo "Musique dataset ready in ./data/raw/"
	@echo "Preparing Musique data (JSONL)..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/prepare_musique_jsonl.py
	@echo "Processed Musique JSONL ready in ./data/processed/questions.jsonl"
	@echo "Extracting unique paragraphs from raw Musique data..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/extract_musique_paragraphs.py
	@echo "Musique paragraphs extracted to ./data/processed/paragraphs.csv"
	@echo "Building Musique FAISS index from paragraphs..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/build_musique_index.py
	@echo "Musique FAISS index files saved to ./data/processed/"
	@echo "Preparing Musique DEV data (JSONL)..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/prepare_musique_dev_jsonl.py
	@echo "Processed Musique DEV JSONL ready in ./data/processed/questions_dev.jsonl"
	@echo "All Musique data preparation steps completed."

## Download ReCall dataset
.PHONY: data-recall
data-recall:
	@echo "Downloading ReCall dataset..."
	$(PYTHON_INTERPRETER) scripts/download/download_recall_data.py
	@echo "ReCall dataset ready in ./data/ReCall-data/"

# Target to run the new simple retrieval server
run-simple-retrieval-server:
	@echo "Starting the simple retrieval server (FlashRAG API Mock) on port 8002..."
	@echo "Access the API docs at http://localhost:8002/docs"
	PYTHONPATH=. uvicorn deploy.serving.serve_simple_retriever:app --host 0.0.0.0 --port 8002 &

# Example: run simple retrieval server with reload for development
run-simple-retrieval-server-dev:
	@echo "Starting the simple retrieval server (FlashRAG API Mock) with reload on port 8002..."
	@echo "Access the API docs at http://localhost:8002/docs"
	PYTHONPATH=. uvicorn deploy.serving.serve_simple_retriever:app --host 0.0.0.0 --port 8002 --reload &

# Target to run the mock retriever server
run-mock-retriever:
	@echo "Starting the mock retriever server on port 8003..."
	PYTHONPATH=. uvicorn deploy.serving.serve_mock_retriever:app --host 0.0.0.0 --port 8003 &

# Target to run the flashrag retriever server
run-flashrag-retriever:
	@echo "Starting the flashrag retriever server on port 8001..."
	PYTHONPATH=. python deploy.serving.serve_flashrag_retriever.py --host 0.0.0.0 --port 8001 --config deploy/serving/retriever_config.yaml &

# Target to run the sandbox_env server
run-sandbox-env:
	@echo "Starting the sandbox_env server on port 8005..."
	PYTHONPATH=. uvicorn deploy.serving.serve_sandbox_env:app --host 0.0.0.0 --port 8005 &

# Target to run all services sequentially without Docker
# Note: This will run them one after the other in the foreground.
# For concurrent execution, you'll need to run each in a separate terminal or background them.
run-all-services-nodocker: run-mock-retriever run-simple-retrieval-server run-flashrag-retriever run-sandbox-env
	@echo "Attempted to start all services in the background."
	@echo "Use 'jobs' to see background processes and 'fg <job_id>' to bring one to foreground, or check 'ps aux | grep uvicorn' and 'ps aux | grep python'."

# Target to kill all non-Docker services
kill-all-services-nodocker:
	@echo "Attempting to kill all non-Docker services..."
	@pkill -f "deploy.serving.serve_mock_retriever" || echo "Mock retriever not found or already stopped."
	@pkill -f "deploy.serving.serve_simple_retriever" || echo "Simple retriever not found or already stopped."
	@pkill -f "deploy.serving.serve_flashrag_retriever.py" || echo "FlashRAG retriever not found or already stopped."
	@pkill -f "deploy.serving.serve_sandbox_env" || echo "Sandbox env not found or already stopped."
	@echo "Kill commands executed. Verify with 'ps aux' or by trying to access the ports."

#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Start all services in detached mode with a build
.PHONY: docker-up
docker-up:
	@echo "Starting all services with Docker Compose..."
	docker compose up --build -d

## Stop all services
.PHONY: docker-down
docker-down:
	@echo "Stopping all services..."
	docker compose down

## View logs for all services
.PHONY: docker-logs
docker-logs:
	@echo "Tailing logs for all services..."
	docker compose logs -f

## Rebuild all service images
.PHONY: docker-build
docker-build:
	@echo "Rebuilding all service images..."
	docker compose build

## Show status of services
.PHONY: docker-ps
docker-ps:
	@echo "Showing status of Docker Compose services..."
	docker compose ps
