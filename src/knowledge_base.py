# src/knowledge_base.py
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from langchain_community.vectorstores import (
    FAISS,
)  # Ensure this is the correct import path for your langchain version

from .config import PROCESSED_DATA_DIR, RAG_SEARCH_RESULTS_COUNT, logger
from .embeddings import CustomHuggingFaceEmbeddings

# Global variable for the vectorstore, explicitly typed
_vectorstore: Optional[FAISS] = None
_questions_data: Optional[List[Dict]] = None


def load_vectorstore(force_reload: bool = False) -> Optional[FAISS]:
    """Load the pre-saved FAISS index."""
    global _vectorstore
    if _vectorstore is not None and not force_reload:
        logger.debug("Vectorstore already loaded. Skipping reload.")
        return _vectorstore

    try:
        embeddings = CustomHuggingFaceEmbeddings()  # Uses default model from config
        # faiss_index_path = PROCESSED_DATA_DIR / "faiss_index"  # Standardize index folder name
        # Corrected path: FAISS index files are directly in PROCESSED_DATA_DIR
        faiss_index_path = PROCESSED_DATA_DIR
        if (
            not faiss_index_path.exists()
        ):  # This check might need adjustment if just a path, not a dir
            logger.error(f"FAISS index directory/path not found at: {faiss_index_path}")
            _vectorstore = None
            return None

        logger.info(f"Loading FAISS index from: {faiss_index_path}")
        _vectorstore = FAISS.load_local(
            str(faiss_index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # Be cautious with this in production
            index_name="index",  # Assuming your files are index.faiss and index.pkl
        )
        logger.info("Successfully loaded FAISS index.")
        return _vectorstore
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}", exc_info=True)
        _vectorstore = None
        return None


def simple_rag_search(
    query: str, k: int = RAG_SEARCH_RESULTS_COUNT
) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks using similarity search.
    Returns a list of dictionaries, each with 'content' and 'metadata'.
    """
    global _vectorstore
    if _vectorstore is None:
        logger.warning("Vectorstore not loaded. Attempting to load now...")
        load_vectorstore()  # Attempt to load if not already
        if _vectorstore is None:  # Still not loaded
            logger.error("Search failed: Vectorstore could not be loaded.")
            return []  # Return empty list if vectorstore is unavailable

    try:
        # similarity_search returns List[Document] where Document has page_content and metadata
        search_results_docs = _vectorstore.similarity_search(query, k=k)

        # Convert to a list of dicts for easier use, including metadata
        results = []
        for doc in search_results_docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
            )
        return results
    except Exception as e:
        logger.error(f"Error during RAG search for query '{query}': {e}", exc_info=True)
        return []


def load_qa_data(
    questions_path: Optional[str] = None, force_reload: bool = False
) -> Optional[List[Dict]]:
    """Load the pre-generated questions from a JSONL file."""
    global _questions_data
    if _questions_data is not None and not force_reload:
        logger.debug("QA data already loaded. Skipping reload.")
        return _questions_data

    try:
        if questions_path is None:
            q_path = PROCESSED_DATA_DIR / "questions.jsonl"
        else:
            q_path = Path(questions_path)

        if not q_path.exists():
            logger.error(f"Questions file not found at: {q_path}")
            _questions_data = None
            return None

        logger.info(f"Loading questions from: {q_path}")
        with open(q_path, "r", encoding="utf-8") as f:
            _questions_data = [json.loads(line) for line in f]
        logger.info(f"Successfully loaded {len(_questions_data)} questions.")
        return _questions_data
    except Exception as e:
        logger.error(f"Error loading QA data: {e}", exc_info=True)
        _questions_data = None
        return None


def get_qa_pair_by_id(qa_id: str) -> Optional[Dict]:
    """Get a specific question-answer pair by its ID."""
    global _questions_data
    if _questions_data is None:
        load_qa_data()
    if _questions_data:
        for qa_item in _questions_data:
            if qa_item.get("id") == qa_id:
                return qa_item
    return None


def get_random_qa_pair() -> Optional[Dict]:
    """Get a random question-answer pair."""
    global _questions_data
    if _questions_data is None:
        load_qa_data()
    if _questions_data:
        return random.choice(_questions_data)
    return None


def get_qa_dataset(
    randomize: bool = False, test_size: float = 0.1, seed: int = 42, questions_path=None
) -> Tuple[Dataset, Dataset]:
    """
    Return HuggingFace Datasets (train, test) containing question and answer pairs.
    """
    global _questions_data
    current_qa_data = _questions_data
    if (
        questions_path is not None or current_qa_data is None
    ):  # Load specific path or if not loaded
        current_qa_data = load_qa_data(questions_path)

    if not current_qa_data:
        logger.error("Cannot create dataset: Questions data not loaded.")
        # Return empty datasets
        return Dataset.from_list([]), Dataset.from_list([])

    qa_dataset = Dataset.from_list(current_qa_data)
    if randomize:
        qa_dataset = qa_dataset.shuffle(seed=seed)

    # Ensure 'prompt' column exists for consistency, renaming 'question'
    if (
        "question" in qa_dataset.column_names
        and "prompt" not in qa_dataset.column_names
    ):
        qa_dataset = qa_dataset.rename_column("question", "prompt")
    elif "prompt" not in qa_dataset.column_names:
        logger.warning(
            "Dataset does not have 'question' or 'prompt' column. Trainer might fail."
        )

    empty_dataset = Dataset.from_list([])
    if test_size <= 0:  # Train only
        return qa_dataset, empty_dataset
    elif test_size >= 1:  # Test only
        return empty_dataset, qa_dataset
    else:  # Split
        split_datasets = qa_dataset.train_test_split(test_size=test_size, seed=seed)
        return split_datasets["train"], split_datasets["test"]


# Initialize vectorstore and QA data on module import, but allow them to be reloaded.
load_vectorstore()
load_qa_data()
