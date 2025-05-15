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

# Target to run the new simple retrieval server
run-simple-retrieval-server:
	@echo "Starting the simple retrieval server (FlashRAG API Mock) on port 8002..."
	@echo "Access the API docs at http://localhost:8002/docs"
	PYTHONPATH=. uvicorn deploy.serving.serve_simple_retriever:app --host 0.0.0.0 --port 8002

# Example: run simple retrieval server with reload for development
run-simple-retrieval-server-dev:
	@echo "Starting the simple retrieval server (FlashRAG API Mock) with reload on port 8002..."
	@echo "Access the API docs at http://localhost:8002/docs"
	PYTHONPATH=. uvicorn deploy.serving.serve_simple_retriever:app --host 0.0.0.0 --port 8002 --reload


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
