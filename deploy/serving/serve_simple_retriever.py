import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path to ensure all imports work correctly
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
print(f"[DEBUG] Added {project_root} to sys.path")
print(f"[DEBUG] Full sys.path: {sys.path}")
print(f"[DEBUG] Current working directory: {os.getcwd()}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

print("[DEBUG] Attempting to import from src...")
# Now try to import directly from src
try:
    from src.config import RAG_SEARCH_RESULTS_COUNT

    print(f"[DEBUG] Successfully imported RAG_SEARCH_RESULTS_COUNT: {RAG_SEARCH_RESULTS_COUNT}")
except ImportError as e:
    print(f"[DEBUG] Failed to import from src.config: {e}")

try:
    from src.knowledge_base import _vectorstore, load_vectorstore, simple_rag_search

    print("[DEBUG] Successfully imported knowledge_base modules")
    print(f"[DEBUG] Initial _vectorstore status: {_vectorstore}")
except ImportError as e:
    print(f"[DEBUG] Failed to import from src.knowledge_base: {e}")

# Global placeholder for the vector store instance (will be set by imports above)
# _vectorstore: object | None = None


# --- Define a base for logging to satisfy linters without importing typing.Protocol ---
class BaseLoggerProtocol:
    def info(self, msg: str) -> None:
        raise NotImplementedError

    def error(self, msg: str, exc_info: bool = False) -> None:
        raise NotImplementedError

    def warning(self, msg: str) -> None:  # Adding warning as it was used
        # Default implementation: print to stdout or delegate to info/error if not overridden
        print(f"WARNING: {msg}")


# --- Define Fallback/Mock components first ---
class SimpleLogger(BaseLoggerProtocol):
    def info(self, msg: str) -> None:
        print(f"INFO: {msg}")

    def error(self, msg: str, exc_info: bool = False) -> None:
        print(f"ERROR: {msg}" + (" (see traceback above)" if exc_info else ""))

    def warning(self, msg: str) -> None:
        print(f"WARNING: {msg}")


logger: BaseLoggerProtocol = SimpleLogger()

# Configuration for the server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("SIMPLE_RETRIEVER_PORT", "8002"))
print(f"[DEBUG] Server config - HOST: {HOST}, PORT: {PORT}")

app = FastAPI(
    title="Simple Retrieval Server (FlashRAG API Mock)",
    description="A simplified server mimicking FlashRAG retriever API with real data.",
    version="0.5.0",  # Incremented version
)


# --- Pydantic Models (FlashRAG compatible) ---
class Document(BaseModel):
    id: str
    contents: str


class QueryRequest(BaseModel):
    query: str
    top_n: int = Field(default=10, gt=0)
    return_score: bool = False


class BatchQueryRequest(BaseModel):
    query: list[str]
    top_n: int = Field(default=10, gt=0)
    return_score: bool = False


def _direct_load_vectorstore() -> None:
    """Directly load the vectorstore without relying on the knowledge_base module."""
    global _vectorstore
    try:
        print("[DEBUG] Trying to load vectorstore directly...")
        from langchain_community.vectorstores import FAISS

        # Try to import the embeddings module using absolute imports
        try:
            from src.embeddings import CustomHuggingFaceEmbeddings

            print("Successfully imported embeddings with absolute import")
        except ImportError as e:
            print(f"[DEBUG] Failed to import embeddings: {e}")
            print("[DEBUG] Falling back to mock")
            _vectorstore = "MockVectorStoreInstanceLoaded"
            return

        print("[DEBUG] Creating embeddings instance")
        embeddings = CustomHuggingFaceEmbeddings()
        print(f"[DEBUG] Embeddings instance created: {type(embeddings)}")

        # Check both absolute and relative paths for index
        potential_paths = [
            "data/processed",  # Relative path (preferred)
            os.path.join(os.getcwd(), "data/processed"),  # Explicit relative path from cwd
            "/home/thinhlpg/code/deep-research/data/processed",  # Legacy absolute path
        ]

        print("[DEBUG] Checking potential index paths:")
        valid_paths = []
        for path in potential_paths:
            print(f"[DEBUG] Checking path: {path}")
            print(f"[DEBUG] Path exists: {os.path.exists(path)}")
            index_file = os.path.join(path, "index.faiss")
            print(f"[DEBUG] Index file: {index_file}")
            print(f"[DEBUG] Index file exists: {os.path.exists(index_file)}")
            if os.path.exists(index_file):
                valid_paths.append(path)
                try:
                    print(f"[DEBUG] Directory contents of {path}:")
                    print(os.listdir(path))
                except Exception as e:
                    print(f"[DEBUG] Failed to list directory: {e}")

        if not valid_paths:
            print("[DEBUG] No valid FAISS index found in any location")
            _vectorstore = "MockVectorStoreInstanceLoaded"
            return

        faiss_index_path = valid_paths[0]
        index_file = os.path.join(faiss_index_path, "index.faiss")

        print(f"[DEBUG] Using faiss_index_path: {faiss_index_path}")
        print(f"[DEBUG] Using index_file: {index_file}")

        if not os.path.exists(index_file):
            print(f"[DEBUG] FAISS index file not found at: {index_file}")
            print(f"[DEBUG] Directory contents of {os.path.dirname(index_file)}:")
            try:
                print(os.listdir(os.path.dirname(index_file)))
            except Exception as e:
                print(f"[DEBUG] Failed to list directory: {e}")
            _vectorstore = "MockVectorStoreInstanceLoaded"
            return

        print(f"[DEBUG] Loading FAISS index from: {faiss_index_path}")
        _vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True,
            index_name="index",
        )
        print(f"[DEBUG] Successfully loaded FAISS index directly: {type(_vectorstore)}")
    except Exception as e:
        print(f"[DEBUG] Error directly loading vectorstore: {e}")
        import traceback

        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        # Fall back to mock data
        _vectorstore = "MockVectorStoreInstanceLoaded"
        print(f"[DEBUG] Fallback to mock: {_vectorstore}")


@app.on_event("startup")
async def startup_event() -> None:
    global _vectorstore
    print("[INFO] Server starting up. Loading vectorstore...")

    try:
        print("[DEBUG] Checking current vectorstore status...")
        if _vectorstore is None:
            print("[DEBUG] Vectorstore not loaded yet. Loading now...")
            print("[DEBUG] Calling load_vectorstore with force_reload=True")

            try:
                # First try to load via the imported function
                print("[DEBUG] Attempting to load via imported load_vectorstore function")
                _vectorstore = load_vectorstore(force_reload=True)
                print(f"[DEBUG] Result from load_vectorstore: {type(_vectorstore)}")
            except Exception as e:
                print(f"[DEBUG] Failed to load via imported function: {e}")
                print("[DEBUG] Falling back to direct loading method")
                _direct_load_vectorstore()

        print(f"[DEBUG] Final vectorstore type: {type(_vectorstore)}")
        if _vectorstore:
            print(f"[DEBUG] Vectorstore loaded successfully: {_vectorstore}")
        else:
            print("[ERROR] Failed to load vectorstore. Server will return 503 errors.")
    except Exception as e:
        print(f"[ERROR] Exception during vectorstore loading: {str(e)}")
        import traceback

        print("[DEBUG] Full startup traceback:")
        traceback.print_exc()


@app.get("/health")
async def health_check() -> dict:  # Changed from dict[str, any] to dict
    available = 1 if _vectorstore is not None else 0
    total = 1  # This simple server always conceptually has one retriever
    status = "healthy" if _vectorstore is not None else "unhealthy"
    print(f"[DEBUG] Health check - status: {status}, vectorstore: {_vectorstore}")
    return {
        "status": status,
        "retrievers": {
            "total": total,
            "available": available,
        },
    }


def _perform_internal_search(query_text: str, top_n_results: int) -> list[dict]:
    """Perform search using the loaded vectorstore."""
    print(f"[DEBUG] Searching for: '{query_text}', top_n={top_n_results}")
    print(f"[DEBUG] Vectorstore type before search: {type(_vectorstore)}")
    try:
        results = simple_rag_search(query=query_text, k=top_n_results)
        print(f"[DEBUG] Search returned {len(results)} results")
        print(f"[DEBUG] First result sample: {results[0] if results else 'None'}")
        return results
    except Exception as e:
        print(f"[DEBUG] Error in _perform_internal_search: {e}")
        import traceback

        traceback.print_exc()
        return []


def _format_results_for_api(
    search_results_list: list[dict],  # list[dict] here too
    return_score_flag: bool,
) -> tuple[list[Document], list[float]] | list[Document]:
    print(f"[DEBUG] Formatting {len(search_results_list)} search results, return_score={return_score_flag}")
    docs_list = []
    scores_list = []
    for i, item_dict in enumerate(search_results_list):
        doc_id_str = str(item_dict.get("metadata", {}).get("doc_id", f"doc_{i}"))
        content_str = str(item_dict.get("content", ""))
        print(f"[DEBUG] Document {i}: id={doc_id_str}, content_length={len(content_str)}")
        docs_list.append(Document(id=doc_id_str, contents=content_str))
        if return_score_flag:
            score_val = float(item_dict.get("metadata", {}).get("score", 1.0))
            scores_list.append(score_val)
            print(f"[DEBUG] Document {i} score: {score_val}")

    if return_score_flag:
        return docs_list, scores_list
    return docs_list


@app.post("/search")
async def search_endpoint(
    request: QueryRequest,
) -> tuple[list[Document], list[float]] | list[Document]:
    print(f"[DEBUG] Received search request: {request}")
    if _vectorstore is None:
        print("[ERROR] Search request failed: Vectorstore not available.")
        raise HTTPException(status_code=503, detail="Vectorstore not available. Please try again later.")
    if not request.query or not request.query.strip():
        print("[ERROR] Search request with empty query.")
        raise HTTPException(status_code=400, detail="Query content cannot be empty.")

    print(f"[INFO] Processing search: '{request.query}', top_n={request.top_n}, return_score={request.return_score}")
    try:
        print("[DEBUG] About to call _perform_internal_search")
        raw_search_results = _perform_internal_search(query_text=request.query, top_n_results=request.top_n)
        print(f"[DEBUG] Got {len(raw_search_results)} raw search results")

        print("[DEBUG] About to format results for API")
        result = _format_results_for_api(
            search_results_list=raw_search_results,
            return_score_flag=request.return_score,
        )
        print(f"[DEBUG] Formatted result type: {type(result)}")
        if isinstance(result, tuple):
            print(f"[DEBUG] Returning {len(result[0])} documents with scores")
        else:
            print(f"[DEBUG] Returning {len(result)} documents without scores")
        return result
    except Exception as e:
        error_msg = f"Error during search for query '{request.query}': {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback

        print("[DEBUG] Full search endpoint traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during the search: {str(e)}")


@app.post("/batch_search")
async def batch_search_endpoint(
    request: BatchQueryRequest,
) -> tuple[list[list[Document]], list[list[float]]] | list[list[Document]]:
    print(f"[DEBUG] Received batch search request with {len(request.query)} queries")
    if _vectorstore is None:
        print("[ERROR] Batch search request failed: Vectorstore not available.")
        raise HTTPException(status_code=503, detail="Vectorstore not available. Please try again later.")
    if not request.query:
        print("[ERROR] Batch search request with empty query list.")
        raise HTTPException(status_code=400, detail="Query list cannot be empty for batch search.")

    print(
        f"[INFO] Processing batch search for {len(request.query)} queries, top_n={request.top_n}, return_score={request.return_score}"
    )
    all_query_docs_results: list[list[Document]] = []
    all_query_scores_results: list[list[float]] = []

    try:
        for i, single_query_text in enumerate(request.query):
            print(f"[DEBUG] Processing batch query {i + 1}/{len(request.query)}: '{single_query_text}'")
            current_query_docs: list[Document] = []
            current_query_scores: list[float] = []
            if not single_query_text or not single_query_text.strip():
                print(
                    f"[WARNING] Empty query string in batch: '{single_query_text}'. Returning empty results for this query."
                )
            else:
                raw_search_results = _perform_internal_search(
                    query_text=single_query_text, top_n_results=request.top_n
                )
                print(f"[DEBUG] Batch query {i + 1} returned {len(raw_search_results)} raw results")

                if request.return_score:
                    print(f"[DEBUG] Formatting batch query {i + 1} results with scores")
                    formatted_result = _format_results_for_api(
                        search_results_list=raw_search_results,
                        return_score_flag=True,
                    )
                    if isinstance(formatted_result, tuple):
                        docs, scores = formatted_result
                        current_query_docs = docs
                        current_query_scores = scores
                else:
                    print(f"[DEBUG] Formatting batch query {i + 1} results without scores")
                    formatted_result = _format_results_for_api(
                        search_results_list=raw_search_results,
                        return_score_flag=False,
                    )
                    # Ensure the correct type handling for the linter error
                    if isinstance(formatted_result, tuple):
                        current_query_docs = formatted_result[0]  # Extract docs from tuple
                    else:
                        current_query_docs = formatted_result

                    # Additional type check
                    if not isinstance(current_query_docs, list):
                        print(f"[DEBUG] Unexpected type for current_query_docs: {type(current_query_docs)}")
                        current_query_docs = []

            print(f"[DEBUG] Adding {len(current_query_docs)} documents to batch results for query {i + 1}")
            all_query_docs_results.append(current_query_docs)
            if request.return_score:
                print(f"[DEBUG] Adding {len(current_query_scores)} scores to batch results for query {i + 1}")
                all_query_scores_results.append(current_query_scores)

        print(f"[DEBUG] Final batch results: {len(all_query_docs_results)} document lists")
        if request.return_score:
            print(f"[DEBUG] Final batch scores: {len(all_query_scores_results)} score lists")
            return all_query_docs_results, all_query_scores_results
        return all_query_docs_results
    except Exception as e:
        error_msg = f"Error during batch search: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback

        print("[DEBUG] Full batch search traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during batch search: {str(e)}")


# --- CLI Handling (Simplified for server only) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Retrieval Server (FlashRAG API Mock)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (ignored by this mock server).",
    )
    parser.add_argument(
        "--num_retriever",
        type=int,
        default=1,
        help="Number of retrievers (ignored, always 1).",
    )
    parser.add_argument("--host", type=str, default=HOST, help=f"Host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    print(f"[DEBUG] Arguments parsed - host: {args.host}, port: {args.port}, config: {args.config}")
    print(f"Starting Simple retrieval server on http://{args.host}:{args.port}")
    if args.config:
        print(f"Note: --config argument ('{args.config}') is ignored by this mock server.")

    try:
        import uvicorn

        print("[DEBUG] Successfully imported uvicorn")
        print(f"[DEBUG] Starting uvicorn server with host={args.host}, port={args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError as e:
        print(f"[DEBUG] Import error for uvicorn: {e}")
        print("ERROR: Uvicorn is not installed. Please run: uv install uvicorn[standard]")
    except Exception as e:
        print(f"ERROR: Failed to start uvicorn server: {e}")
        import traceback

        print("[DEBUG] Server startup error traceback:")
        traceback.print_exc()
