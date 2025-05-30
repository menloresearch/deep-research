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
	ruff check --fix
	ruff format


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
