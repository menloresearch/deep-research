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

#################################################################################
# DATA PREPARATION                                                              #
#################################################################################

## Prepare all data (Musique and ReCall)
.PHONY: data
data: prepare-data download-recall-syntool
	@echo "All data preparation completed"

## Download and prepare all Musique data
.PHONY: prepare-data
prepare-data: download-musique prepare-musique convert-musique-recall-format

## Download Musique dataset
.PHONY: download-musique
download-musique:
	@echo "Downloading Musique dataset..."
	bash scripts/prepare_train_data/01_download_data_musique.sh
	@echo "Musique dataset ready in ./data/raw/"

## Prepare Musique data
.PHONY: prepare-musique
prepare-musique:
	@echo "Preparing Musique data..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/02_prepare_musique_jsonl.py
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/03_extract_musique_paragraphs.py
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/04_build_musique_index.py
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/05_prepare_musique_dev_jsonl.py
	@echo "Musique data preparation completed."

## Convert Musique data to ReCall format
.PHONY: convert-musique-recall-format
convert-musique-recall-format:
	@echo "Converting Musique data to ReCall format..."
	$(PYTHON_INTERPRETER) scripts/prepare_train_data/06_convert_musique_recall_format.py --input_dir data/processed --output_dir data/processed
	@echo "Musique data conversion to ReCall format completed."

## Download and sample ReCall syntool data
.PHONY: download-recall-syntool
download-recall-syntool:
	@echo "Downloading ReCall syntool data from HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/download/download_data.py --repo-id thinhlp/ReCall-data --output-dir data/ReCall-data --repo-type dataset
	@echo "Creating sampled subset of syntool_re_call data..."
	$(PYTHON_INTERPRETER) scripts/download/sample_recall_data.py --input-dir data/ReCall-data/syntool_re_call --output-dir data/ReCall-data/syntool_re_call_sampled --train-samples 1000 --test-samples 50
	@echo "ReCall syntool data sampled and saved to data/ReCall-data/syntool_re_call_sampled"

#################################################################################
# DATA TRANSFER                                                                 #
#################################################################################

## Download data from HuggingFace
.PHONY: download-data
download-data:
	@echo "Downloading full data from HuggingFace to data/ directory..."
	$(PYTHON_INTERPRETER) scripts/download/download_data.py --type data --output-dir data
	@echo "Full dataset downloaded successfully to data/"

## Download model from HuggingFace
.PHONY: download-model
download-model:
	@echo "Downloading model from HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/download/download_data.py --type model --output-dir data
	@echo "Model downloaded successfully"

## Download checkpoint from HuggingFace
.PHONY: download-checkpoint
download-checkpoint:
	@echo "Downloading checkpoint from HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/download/download_data.py --type model --output-dir checkpoints
	@echo "Checkpoint downloaded successfully"

## Upload data to HuggingFace
.PHONY: upload-data
upload-data:
	@echo "Uploading full data/ directory to HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/upload/upload_data.py --type data --dir data
	@echo "Full data directory uploaded successfully"

## Upload model to HuggingFace
.PHONY: upload-model
upload-model:
	@echo "Uploading model to HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/upload/upload_data.py --type model --dir data
	@echo "Model uploaded successfully"

## Upload checkpoint to HuggingFace
.PHONY: upload-checkpoint
upload-checkpoint:
	@echo "Uploading checkpoint to HuggingFace..."
	$(PYTHON_INTERPRETER) scripts/upload/upload_data.py --type model --dir checkpoints
	@echo "Checkpoint uploaded successfully"

#################################################################################
# SERVING                                                                       #
#################################################################################

## Start vLLM server
.PHONY: serve-vllm
serve-vllm:
	@echo "Starting vLLM server using scripts/run_vllm_server.sh..."
	@echo "You can customize by setting environment variables like CUDA_VISIBLE_DEVICES, VLLM_MODEL, VLLM_TP_SIZE, etc."
	PYTHONPATH=. bash scripts/run_vllm_server.sh &

## Start all services
.PHONY: serve-all
serve-all: serve-mock serve-simple serve-flashrag serve-sandbox serve-vllm

## Start mock retriever
.PHONY: serve-mock
serve-mock:
	@echo "Starting mock retriever using scripts/run_mock_retriever_server.sh..."
	PYTHONPATH=. bash scripts/run_mock_retriever_server.sh &

## Start simple retrieval server
.PHONY: serve-simple
serve-simple:
	@echo "Starting simple retrieval server using scripts/run_simple_retriever_server.sh..."
	PYTHONPATH=. bash scripts/run_simple_retriever_server.sh &

## Start flashrag retriever
.PHONY: serve-flashrag
serve-flashrag:
	@echo "Starting flashrag retriever using scripts/run_flashrag_server.sh..."
	PYTHONPATH=. bash scripts/run_flashrag_server.sh &

## Start sandbox environment
.PHONY: serve-sandbox
serve-sandbox:
	@echo "Starting sandbox environment using scripts/run_sandbox_server.sh..."
	PYTHONPATH=. bash scripts/run_sandbox_server.sh &

## Start all services (non-docker)
.PHONY: run-all-services-nodocker
run-all-services-nodocker: serve-all
	@echo "All services started in non-docker mode"

## Stop all services
.PHONY: stop-all
stop-all:
	@echo "Stopping all services..."
	@-pkill -f "bash scripts/run_mock_retriever_server.sh" || echo "No mock retriever script running."
	@-pkill -f "uvicorn deploy.serving.serve_mock_retriever:app" || echo "No mock retriever uvicorn process running."
	@-pkill -f "bash scripts/run_simple_retriever_server.sh" || echo "No simple retriever script running."
	@-pkill -f "uvicorn deploy.serving.serve_simple_retriever:app" || echo "No simple retriever uvicorn process running."
	@-pkill -f "bash scripts/run_flashrag_server.sh" || echo "No FlashRAG retriever script running."
	@-pkill -f "python deploy.serving.serve_flashrag_retriever.py" || echo "No FlashRAG retriever python process running."
	@-pkill -f "bash scripts/run_sandbox_server.sh" || echo "No sandbox env script running."
	@-pkill -f "uvicorn deploy.serving.serve_sandbox_env:app" || echo "No sandbox env uvicorn process running."
	@-pkill -f "bash scripts/run_vllm_server.sh" || echo "No vLLM server script running."
	@-pkill -f "python verifiers-deepresearch/verifiers/inference/vllm_serve.py" || echo "No vLLM server python process running."
	@-pkill -9 -f "deploy/serving/serve" || echo "No other top-level serving processes found."
	@-lsof -ti:8000,8001,8002,8003,8005 | xargs kill -9 2>/dev/null || echo "No processes on known service ports."
	@echo "All services should be stopped now."

## Stop all services (alias to stop-all)
.PHONY: kill-all-services-nodocker
kill-all-services-nodocker: stop-all
	@echo "All services stopped in non-docker mode"

#################################################################################
# DOCKER                                                                        #
#################################################################################

## Start all services with Docker
.PHONY: docker-up
docker-up:
	@echo "Starting all services with Docker Compose..."
	docker compose up --build -d

## Stop all Docker services
.PHONY: docker-down
docker-down:
	@echo "Stopping all services..."
	docker compose down

## View Docker logs
.PHONY: docker-logs
docker-logs:
	@echo "Tailing logs for all services..."
	docker compose logs -f

## Rebuild Docker images
.PHONY: docker-build
docker-build:
	@echo "Rebuilding all service images..."
	docker compose build

## Show Docker service status
.PHONY: docker-ps
docker-ps:
	@echo "Showing status of Docker Compose services..."
	docker compose ps
