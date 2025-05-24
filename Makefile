# Model conversion settings
SOURCE_MODEL = jan-hq/Qwen3-14B-v0.1-deepresearch-100-step
TARGET_REPO = janhq/Qwen3-14B-v0.1-deepresearch-100-step-gguf
TEMP_DIR = temp_model
GGUF_DIR = gguf_output

# Default target
.PHONY: gguf-all
gguf-all: gguf-setup gguf-build gguf-download gguf-convert gguf-upload gguf-clean

# Setup environment and dependencies
.PHONY: gguf-setup
gguf-setup:
	@echo "Setting up GGUF environment..."
	uv pip install huggingface_hub transformers torch python-dotenv
	git clone https://github.com/ggerganov/llama.cpp.git || true
	uv pip install -r llama.cpp/requirements.txt --index-strategy unsafe-best-match || \
	uv pip install numpy scipy gguf || echo "Installing basic requirements..."
	@echo "✓ GGUF setup complete"

# Build llama.cpp binaries
.PHONY: gguf-build
gguf-build:
	@echo "Building llama.cpp binaries..."
	cd llama.cpp && mkdir -p build && cd build && cmake .. && cmake --build . --config Release --target llama-quantize
	@echo "✓ llama-quantize binary built"

# Download the source model
.PHONY: gguf-download
gguf-download:
	@echo "Downloading model: $(SOURCE_MODEL)"
	python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$(SOURCE_MODEL)', local_dir='$(TEMP_DIR)', local_dir_use_symlinks=False)"
	@echo "✓ Model downloaded to $(TEMP_DIR)"

# Convert to GGUF format using the Python script
.PHONY: gguf-convert
gguf-convert:
	@echo "Converting to GGUF format..."
	mkdir -p $(GGUF_DIR)
	python utils/98_convert_gguf.py $(TEMP_DIR) $(GGUF_DIR)
	@echo "✓ GGUF conversion complete"

# Create and upload to HuggingFace
.PHONY: gguf-upload
gguf-upload:
	@echo "Creating target repository and uploading..."
	python utils/99_upload.py --local-dir $(GGUF_DIR) --repo-id $(TARGET_REPO)
	@echo "✓ Uploaded to https://huggingface.co/$(TARGET_REPO)"

# Clean up temporary files
.PHONY: gguf-clean
gguf-clean:
	@echo "Cleaning up GGUF files..."
	rm -rf $(TEMP_DIR) $(GGUF_DIR) llama.cpp
	@echo "✓ GGUF cleanup complete"

# Help target
.PHONY: gguf-help
gguf-help:
	@echo "Available GGUF conversion targets:"
	@echo "  gguf-all      - Complete workflow: setup, build, download, convert, upload, clean"
	@echo "  gguf-setup    - Install dependencies and clone llama.cpp"
	@echo "  gguf-build    - Build llama.cpp binaries (llama-quantize)"
	@echo "  gguf-download - Download the source model"
	@echo "  gguf-convert  - Convert model to GGUF format"
	@echo "  gguf-upload   - Upload GGUF model to HuggingFace"
	@echo "  gguf-clean    - Clean up temporary files"
	@echo "  gguf-help     - Show this help message"
	@echo ""
	@echo "Model: $(SOURCE_MODEL) -> $(TARGET_REPO)" 