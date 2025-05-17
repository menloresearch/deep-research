import sys
from pathlib import Path

# Add the project root to sys.path to allow importing from 'src'
# __file__ is 'deep-research/scripts/prepare_train_data/build_musique_index.py'
# project_root should be 'deep-research'
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import FAISS after potentially adding to sys.path
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    print("Error: langchain_community or FAISS not installed. Please install with 'uv install langchain faiss-cpu'")
    sys.exit(1)

import json
import math  # Import math for ceiling division
import traceback  # Import traceback

import pandas as pd

from src.embeddings import CustomHuggingFaceEmbeddings

# Constants
CONTENT_COLUMN = "content"
METADATA_COLUMN = "metadata"
DEFAULT_BATCH_SIZE = 512
DEFAULT_INPUT_CSV_NAME = "paragraphs.csv"
PROCESSED_DATA_DIR_NAME = "data/processed"


def load_and_validate_dataframe(csv_path: Path) -> pd.DataFrame | None:
    """Loads and validates the DataFrame from a CSV file."""
    print(f"Loading paragraphs from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please run the extraction script first.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    if CONTENT_COLUMN not in df.columns or METADATA_COLUMN not in df.columns:
        print(f"Error: CSV file must contain '{CONTENT_COLUMN}' and '{METADATA_COLUMN}' columns.")
        return None

    if df.empty:
        print("Warning: Input CSV file is empty. No index will be built.")
        return df
    return df


def prepare_documents_from_dataframe(
    df: pd.DataFrame,
) -> tuple[list[str], list[dict]] | None:
    """Prepares texts and metadatas from the DataFrame."""
    if df.empty:  # Handle case where df is empty but valid
        return [], []

    # Explicitly convert each item to string in a list comprehension
    texts: list[str] = [str(item) for item in df[CONTENT_COLUMN].tolist()]
    metadatas: list[dict] = []
    try:
        metadatas = [json.loads(str(m)) for m in df[METADATA_COLUMN].tolist()]
        print(f"Prepared {len(texts)} texts and {len(metadatas)} metadatas.")
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}. Check the format in the CSV.")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error processing metadata: {e}")
        traceback.print_exc()
        return None

    if not texts or not metadatas or len(texts) != len(metadatas):
        print(f"Error: Mismatch or empty texts/metadatas. Texts: {len(texts)}, Metadatas: {len(metadatas)}")
        return None
    return texts, metadatas


def build_faiss_index(csv_path: Path, index_save_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Builds a FAISS index from a CSV containing paragraph content and metadata."""
    df = load_and_validate_dataframe(csv_path)
    if df is None:  # Error occurred during loading/validation
        return
    if df.empty:  # CSV was empty, but valid
        print("Input CSV is empty. Skipping index creation.")
        return

    prepared_docs = prepare_documents_from_dataframe(df)
    if prepared_docs is None:
        return
    texts, metadatas = prepared_docs

    if not texts:  # Should be caught by df.empty earlier, but as a safeguard
        print("No texts to process after preparation. Skipping index creation.")
        return

    print("Initializing embeddings model...")
    try:
        embeddings = CustomHuggingFaceEmbeddings()
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        traceback.print_exc()
        return
    print("Embeddings model initialized successfully.")

    vectorstore: FAISS | None = None  # Explicitly type vectorstore
    num_batches = math.ceil(len(texts) / batch_size)
    print(f"Processing {len(texts)} texts in {num_batches} batches of size {batch_size}...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        print(f"  Processing batch {i + 1}/{num_batches} (indices {start_idx}-{end_idx - 1})...")

        try:
            if i == 0:
                print("    Initializing FAISS index with first batch...")
                vectorstore = FAISS.from_texts(texts=batch_texts, embedding=embeddings, metadatas=batch_metadatas)
                print("    FAISS index initialized.")
            else:
                if vectorstore is None:  # Should not happen if first batch succeeded
                    print("Error: vectorstore is None after first batch. Halting.")
                    return
                print(f"    Adding batch {i + 1} to FAISS index...")
                vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                print(f"    Batch {i + 1} added.")
        except Exception as e:
            print(f"Error processing batch {i + 1} (indices {start_idx}-{end_idx - 1}): {e}")
            traceback.print_exc()
            print("Stopping index creation due to error in batch processing.")
            return

    if vectorstore is None:
        print("Error: Failed to create or add any data to the vectorstore.")
        return

    try:
        print(f"Attempting to save final FAISS index files to directory: {index_save_path}")
        index_save_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_save_path))  # save_local expects a string path
        print(f"Successfully saved final FAISS index files (index.faiss, index.pkl) to: {index_save_path}")
    except Exception as e:
        print(f"Error during final vectorstore.save_local to {index_save_path}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Define paths relative to the project root
    current_project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = current_project_root / PROCESSED_DATA_DIR_NAME
    input_csv_path = processed_dir / DEFAULT_INPUT_CSV_NAME
    # FAISS save_local will save index.faiss and index.pkl in this directory
    # Save directly to the processed_dir, so index files are alongside paragraphs.csv
    faiss_index_save_dir = processed_dir

    build_faiss_index(input_csv_path, faiss_index_save_dir, batch_size=DEFAULT_BATCH_SIZE)
