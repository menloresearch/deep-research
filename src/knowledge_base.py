# src/knowledge_base.py
import json
import os
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

print(f"[DEBUG] KB INIT: Module initialized, _vectorstore={_vectorstore}")
print(f"[DEBUG] KB INIT: PROCESSED_DATA_DIR={PROCESSED_DATA_DIR}")


def load_vectorstore(force_reload: bool = False) -> Optional[FAISS]:
    """Load the pre-saved FAISS index."""
    global _vectorstore
    if _vectorstore is not None and not force_reload:
        logger.debug("Vectorstore already loaded. Skipping reload.")
        print(f"[DEBUG] KB: Vectorstore already loaded. Skipping reload. Type: {type(_vectorstore)}")
        return _vectorstore

    try:
        print(f"[DEBUG] KB: Starting vectorstore loading process (force_reload={force_reload})")
        print(f"[DEBUG] KB: Current working directory: {os.getcwd()}")

        try:
            embeddings = CustomHuggingFaceEmbeddings()  # Uses default model from config
            print(f"[DEBUG] KB: Successfully created embeddings instance: {type(embeddings)}")
        except Exception as e:
            print(f"[DEBUG] KB: Error creating embeddings: {e}")
            import traceback

            traceback.print_exc()
            _vectorstore = None
            return None

        # Try relative path first, then fall back to alternatives
        potential_paths = [
            "data/processed",  # Relative path (preferred)
            str(PROCESSED_DATA_DIR),  # From config
            "/home/thinhlpg/code/deep-research/data/processed",  # Legacy absolute path as fallback
        ]

        print("[DEBUG] KB: Checking all potential index paths:")
        valid_paths = []
        for path in potential_paths:
            index_file = os.path.join(path, "index.faiss")
            pickle_file = os.path.join(path, "index.pkl")
            print(f"[DEBUG] KB: Checking path: {path}")
            print(f"[DEBUG] KB: Path exists: {os.path.exists(path)}")
            print(f"[DEBUG] KB: index.faiss exists: {os.path.exists(index_file)}")
            print(f"[DEBUG] KB: index.pkl exists: {os.path.exists(pickle_file)}")
            if os.path.exists(index_file) and os.path.exists(pickle_file):
                print(f"[DEBUG] KB: Found valid index at: {path}")
                valid_paths.append(path)
                try:
                    print(f"[DEBUG] KB: Directory contents of {path}:")
                    print(os.listdir(path))
                except Exception as e:
                    print(f"[DEBUG] KB: Failed to list directory: {e}")

        # Use the first valid path found (prioritizing relative paths)
        if not valid_paths:
            error_msg = "No valid FAISS index found in any of the potential paths"
            logger.error(error_msg)
            print(f"[ERROR] KB: {error_msg}")
            _vectorstore = None
            return None

        faiss_index_path = valid_paths[0]
        index_file = os.path.join(faiss_index_path, "index.faiss")
        index_pickle = os.path.join(faiss_index_path, "index.pkl")

        print(f"[DEBUG] KB: Selected index path: {faiss_index_path}")
        print(f"[DEBUG] KB: Using index file: {index_file}")
        print(f"[DEBUG] KB: Using pickle file: {index_pickle}")

        print(f"[DEBUG] KB: Loading FAISS index from: {faiss_index_path}")
        logger.info(f"Loading FAISS index from: {index_file}")

        print(f"[DEBUG] KB: Using embeddings model type: {type(embeddings)}")
        print(f"[DEBUG] KB: Calling FAISS.load_local with path: {faiss_index_path}, index_name: index")

        try:
            _vectorstore = FAISS.load_local(
                faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True,  # Be cautious with this in production
                index_name="index",  # Assuming your files are index.faiss and index.pkl
            )
            print(f"[DEBUG] KB: FAISS loaded successfully, vectorstore type: {type(_vectorstore)}")
            logger.info("Successfully loaded FAISS index.")
            return _vectorstore
        except Exception as e:
            print(f"[DEBUG] KB: Error in FAISS.load_local: {e}")
            import traceback

            traceback.print_exc()
            _vectorstore = None
            return None
    except Exception as e:
        error_msg = f"Error loading vectorstore: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"[ERROR] KB: {error_msg}")
        import traceback

        traceback.print_exc()
        _vectorstore = None
        return None


def simple_rag_search(query: str, k: int = RAG_SEARCH_RESULTS_COUNT) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks using similarity search.
    Returns a list of dictionaries, each with 'content' and 'metadata'.
    """
    global _vectorstore
    print(f"[DEBUG] KB search: Starting search for '{query}', k={k}")
    print(f"[DEBUG] KB search: Current vectorstore: {_vectorstore}")

    if _vectorstore is None:
        logger.warning("Vectorstore not loaded. Attempting to load now...")
        print("[DEBUG] KB search: Vectorstore not loaded. Attempting to load now...")
        _vectorstore = load_vectorstore()  # Attempt to load if not already
        print(f"[DEBUG] KB search: After load attempt, vectorstore: {_vectorstore}")

        if _vectorstore is None:  # Still not loaded
            logger.error("Search failed: Vectorstore could not be loaded.")
            print("[ERROR] KB search: Search failed: Vectorstore could not be loaded.")
            return []  # Return empty list if vectorstore is unavailable

    try:
        # Check if vectorstore is a mockup
        if isinstance(_vectorstore, str):
            print(f"[DEBUG] KB search: Using mock vectorstore: {_vectorstore}")
            # Return mock results
            return [
                {
                    "content": f"Mock search result for query: {query}",
                    "metadata": {"doc_id": "mock_1", "score": 0.95},
                }
            ] * min(k, 5)  # Return at most 5 mock results

        print("[DEBUG] KB search: Using real vectorstore for similarity_search")
        # similarity_search returns List[Document] where Document has page_content and metadata
        search_results_docs = _vectorstore.similarity_search(query, k=k)
        print(f"[DEBUG] KB search: similarity_search returned {len(search_results_docs)} results")

        # Convert to a list of dicts for easier use, including metadata
        results = []
        for i, doc in enumerate(search_results_docs):
            print(f"[DEBUG] KB search: Processing result {i + 1}/{len(search_results_docs)}")
            result_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
            print(f"[DEBUG] KB search: Result {i + 1} content length: {len(result_dict['content'])}")
            print(f"[DEBUG] KB search: Result {i + 1} metadata: {result_dict['metadata']}")
            results.append(result_dict)

        print(f"[DEBUG] KB search: Returning {len(results)} processed results")
        return results
    except Exception as e:
        logger.error(f"Error during RAG search for query '{query}': {e}", exc_info=True)
        print(f"[ERROR] KB search: Error during search: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_qa_data(questions_path: Optional[str] = None, force_reload: bool = False) -> Optional[List[Dict]]:
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
    if questions_path is not None or current_qa_data is None:  # Load specific path or if not loaded
        current_qa_data = load_qa_data(questions_path)

    if not current_qa_data:
        logger.error("Cannot create dataset: Questions data not loaded.")
        # Return empty datasets
        return Dataset.from_list([]), Dataset.from_list([])

    qa_dataset = Dataset.from_list(current_qa_data)
    if randomize:
        qa_dataset = qa_dataset.shuffle(seed=seed)

    # Ensure 'prompt' column exists for consistency, renaming 'question'
    if "question" in qa_dataset.column_names and "prompt" not in qa_dataset.column_names:
        qa_dataset = qa_dataset.rename_column("question", "prompt")
    elif "prompt" not in qa_dataset.column_names:
        logger.warning("Dataset does not have 'question' or 'prompt' column. Trainer might fail.")

    empty_dataset = Dataset.from_list([])
    if test_size <= 0:  # Train only
        return qa_dataset, empty_dataset
    elif test_size >= 1:  # Test only
        return empty_dataset, qa_dataset
    else:  # Split
        split_datasets = qa_dataset.train_test_split(test_size=test_size, seed=seed)
        return split_datasets["train"], split_datasets["test"]


# Initialize vectorstore and QA data on module import, but allow them to be reloaded.
print("[DEBUG] KB INIT: About to call initial load_vectorstore()")
load_vectorstore()
print("[DEBUG] KB INIT: Initial vectorstore loaded, status:", "loaded" if _vectorstore else "failed")
load_qa_data()
