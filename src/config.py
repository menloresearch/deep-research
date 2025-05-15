# src/config.py
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


# --- Basic Logging Setup (Called once on import) ---
def _setup_basic_logging(log_folder: Path):
    """Sets up a basic console and file logger."""
    log_folder.mkdir(parents=True, exist_ok=True)
    logger.remove()  # Remove any default handlers

    # Console logger
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level=os.getenv("LOG_LEVEL", "INFO").upper(),  # Control via LOG_LEVEL env var
        colorize=True,
        backtrace=True,  # Useful for development
        diagnose=True,  # Useful for development
    )

    # General file logger
    logger.add(
        log_folder / "app.log",  # Simple, general log file
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # Log more details to file
        rotation="50 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,  # For async safety
    )

    # Custom log level example (optional)
    logger.level("REQUEST", no=35, color="<yellow>", icon="🚀")

    # Basic exception hook for unhandled exceptions
    def basic_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)  # Default for Ctrl+C
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "Unhandled critical exception:"
        )

    sys.excepthook = basic_exception_handler
    logger.info("Basic logging configured.")


# --- Load Environment Variables ---
load_dotenv(override=True)  # Load .env file if present

# --- Core Project Paths ---
SRC_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SRC_DIR.parent
DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"  # Directory for storing models
LOG_DIR = PROJ_ROOT / "logs"  # General log directory

# --- Initialize Basic Logging ---
_setup_basic_logging(LOG_DIR)
logger.info(f"Project Root: {PROJ_ROOT}")
logger.info(f"Log Directory: {LOG_DIR}")

# --- Agent & Application Specific Configurations ---
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # For FAISS index, questions.jsonl etc.

# Special Tokens for Agent Interaction
BEGIN_THINK = "<think>"
END_THINK = "</think>"
BEGIN_SEARCH = "<search>"
END_SEARCH = "</search>"
BEGIN_ANSWER = "<answer>"
END_ANSWER = "</answer>"
INFORMATION_START = "<information>"
INFORMATION_END = "</information>"

BEGIN_WEB_SEARCH = "<web_search>"
END_WEB_SEARCH = "</web_search>"
BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"
BEGIN_WRITE_SECTION = "<|begin_write_section|>"
END_WRITE_SECTION = "<|end_write_section|>"
# Add other tokens if needed: BEGIN_EDIT_ARTICLE, BEGIN_CHECK_ARTICLE, BEGIN_SEARCH_RESULT etc.

# API Keys (from .env)
BING_API_KEY = os.getenv("BING_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # Retain for other potential OpenAI direct uses or legacy
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For OpenRouter

# OpenRouter Specific Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default LLM and Agent Configuration
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen/qwen3-32b")  # Changed default
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "intfloat/e5-base-v2")
MAX_CONTEXT_LENGTH_LLM = int(os.getenv("MAX_CONTEXT_LENGTH_LLM", 32768))
MAX_GENERATION_TOKENS = int(os.getenv("MAX_GENERATION_TOKENS", 1024))
AGENT_MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", 20))

# Search Configuration for Tools
RAG_SEARCH_RESULTS_COUNT = int(os.getenv("RAG_SEARCH_RESULTS_COUNT", 5))
WEB_SEARCH_RESULTS_COUNT = int(os.getenv("WEB_SEARCH_RESULTS_COUNT", 5))

# Model Server configurations (if you still run separate model servers for retriever/generator)
# These are simplified and assume you might still use them for inference.
RETRIEVER_MODEL_REPO_ID = DEFAULT_EMBEDDING_MODEL
GENERATOR_MODEL_REPO_ID = os.getenv(
    "GENERATOR_MODEL_REPO_ID", DEFAULT_LLM_MODEL
)  # Or a specific generator model

# --- Minimal Torch import (optional, if not used directly in config) ---
# If you don't need torch here, you can remove it.
# import torch
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# logger.debug(f"PyTorch device availability: {DEVICE}")


if __name__ == "__main__":
    logger.info("Config script executed directly (for testing).")
    logger.debug(f"OpenAI API Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
    logger.debug(f"Default Agent LLM: {DEFAULT_LLM_MODEL}")
    logger.info("This is an info message.")
    logger.debug("This is a debug message (visible if LOG_LEVEL=DEBUG).")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.critical("A critical (handled) error occurred during config test.")
    logger.info("Config test finished.")
