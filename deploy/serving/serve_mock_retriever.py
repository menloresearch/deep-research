import argparse
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Global placeholder for the vector store instance
_vectorstore: object | None = None


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
RAG_SEARCH_RESULTS_COUNT: int = 3


def _initial_mock_load_vectorstore() -> None:
    global _vectorstore
    # logger.info("Mock load_vectorstore called. Setting mock vectorstore.") # Removed log
    print("Mock load_vectorstore called. Setting mock vectorstore.")  # Replaced with print
    _vectorstore = "MockVectorStoreInstanceLoaded"


def _initial_mock_simple_rag_search(query: str, k: int) -> list[dict]:
    # logger.info(f"Mock simple_rag_search called with query: '{query}', k: {k}") # Removed log
    if _vectorstore is None:
        # logger.error("Mock search called but mock vectorstore is not loaded.") # Removed log
        print("ERROR: Mock search called but mock vectorstore is not loaded.")  # Replaced with print
        return []
    results = []
    for i in range(min(k, 5)):  # Return at most 5 mock results, or k if less
        results.append(
            {
                "content": f"Mock search result {i + 1} for query: {query}",
                "metadata": {
                    "doc_id": f"mock_doc_id_{i + 1}_for_{query[:10]}",
                    "source": "mock_source_data",
                    "score": 0.95 - (i * 0.05),  # Mock score
                },
            }
        )
    return results


# Initialize with fallbacks. These will be overwritten if real imports succeed.
# logger removed
load_vectorstore = _initial_mock_load_vectorstore
simple_rag_search = _initial_mock_simple_rag_search

try:
    from ...src.config import RAG_SEARCH_RESULTS_COUNT as _real_rag_count

    # from ...src.config import logger as _real_logger_obj # Logger import removed
    from ...src.knowledge_base import _vectorstore as _kb_vectorstore
    from ...src.knowledge_base import load_vectorstore as _real_load_vectorstore
    from ...src.knowledge_base import simple_rag_search as _real_simple_rag_search

    _vectorstore = _kb_vectorstore
    load_vectorstore = _real_load_vectorstore
    simple_rag_search = _real_simple_rag_search
    # logger = _real_logger_obj # Logger assignment removed
    RAG_SEARCH_RESULTS_COUNT = _real_rag_count
    # logger.info("Successfully imported real components.") # Removed log
    print("Successfully imported real components.")  # Replaced with print

except ImportError:
    print("Warning: Running with standalone script. Relative imports failed. Using mock components.")
    print("Ensure PYTHONPATH is set correctly (e.g., to project root) if expecting real components.")
    # Fallbacks are already set.

# Configuration for the server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("MOCK_RETRIEVER_PORT", "8003"))

app = FastAPI(
    title="Mock Retrieval Server (FlashRAG API)",
    description="A mock server mimicking FlashRAG retriever API with predefined mock data.",
    version="1.0.0",
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


@app.on_event("startup")
async def startup_event() -> None:
    # logger.info("Mock Server starting up. Loading mock vectorstore...") # Removed log
    print("Mock Server starting up. Loading mock vectorstore...")  # Replaced with print
    load_vectorstore()  # Load the mock vectorstore
    if _vectorstore:
        # logger.info("Mock vectorstore initialized successfully.") # Removed log
        print("Mock vectorstore initialized successfully.")  # Replaced with print
    else:
        # logger.error("Mock vectorstore failed to initialize.") # Removed log
        print("ERROR: Mock vectorstore failed to initialize.")  # Replaced with print


@app.get("/health")
async def health_check() -> dict:
    available = 1 if _vectorstore is not None else 0
    total = 1  # This mock server always conceptually has one retriever
    status = "healthy" if _vectorstore is not None else "unhealthy"
    return {
        "status": status,
        "retrievers": {
            "total": total,
            "available": available,
        },
    }


def _perform_internal_search(query_text: str, top_n_results: int) -> list[dict]:
    return simple_rag_search(query=query_text, k=top_n_results)


def _format_results_for_api(
    search_results_list: list[dict],
    return_score_flag: bool,
) -> tuple[list[Document], list[float]] | list[Document]:
    docs_list = []
    scores_list = []
    for i, item_dict in enumerate(search_results_list):
        metadata = item_dict.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        doc_id_str = str(metadata.get("doc_id", f"fallback_doc_id_{i}"))
        content_str = str(item_dict.get("content", ""))
        docs_list.append(Document(id=doc_id_str, contents=content_str))

        if return_score_flag:
            score_val = float(metadata.get("score", 1.0))
            scores_list.append(score_val)

    if return_score_flag:
        return docs_list, scores_list
    return docs_list


@app.post("/search")
async def search_endpoint(
    request: QueryRequest,
) -> tuple[list[Document], list[float]] | list[Document]:
    if _vectorstore is None:
        # logger.error("Search request failed: Mock Vectorstore not available.") # Removed log
        raise HTTPException(status_code=503, detail="Mock Vectorstore not available.")
    if not request.query or not request.query.strip():
        # logger.warning("Search request with empty query.") # Removed log
        raise HTTPException(status_code=400, detail="Query content cannot be empty.")

    # logger.info(f"MOCK Processing search: '{request.query}', top_n={request.top_n}, return_score={request.return_score}") # Removed log
    print(
        f"MOCK Processing search: '{request.query}', top_n={request.top_n}, return_score={request.return_score}"
    )  # Replaced with print
    try:
        raw_search_results = _perform_internal_search(query_text=request.query, top_n_results=request.top_n)
        return _format_results_for_api(
            search_results_list=raw_search_results,
            return_score_flag=request.return_score,
        )
    except Exception as e:
        # logger.error(f"Error during MOCK search for query '{request.query}': {e}", exc_info=True) # Removed log
        print(f"Error during MOCK search for query '{request.query}': {e}")  # Replaced with print
        raise HTTPException(status_code=500, detail="An error occurred during the mock search.")


@app.post("/batch_search")
async def batch_search_endpoint(
    request: BatchQueryRequest,
) -> tuple[list[list[Document]], list[list[float]]] | list[list[Document]]:
    if _vectorstore is None:
        # logger.error("Batch search request failed: Mock Vectorstore not available.") # Removed log
        raise HTTPException(status_code=503, detail="Mock Vectorstore not available.")
    if not request.query:
        # logger.warning("Batch search request with empty query list.") # Removed log
        raise HTTPException(status_code=400, detail="Query list cannot be empty for batch search.")

    # logger.info(f"MOCK Processing batch search for {len(request.query)} queries, top_n={request.top_n}, return_score={request.return_score}") # Removed log
    print(
        f"MOCK Processing batch search for {len(request.query)} queries, top_n={request.top_n}, return_score={request.return_score}"
    )  # Replaced with print
    all_query_docs_results: list[list[Document]] = []
    all_query_scores_results: list[list[float]] = []

    try:
        for single_query_text in request.query:
            current_query_docs: list[Document]
            current_query_scores: list[float]

            if not single_query_text or not single_query_text.strip():
                # logger.warning(f"Empty query string in batch: '{single_query_text}'. Returning empty results for this query.") # Removed log
                print(
                    f"Warning: Empty query string in batch: '{single_query_text}'. Returning empty results for this query."
                )  # Replaced with print
                current_query_docs = []
                current_query_scores = []
            else:
                raw_search_results = _perform_internal_search(
                    query_text=single_query_text, top_n_results=request.top_n
                )
                formatted_output = _format_results_for_api(
                    search_results_list=raw_search_results,
                    return_score_flag=request.return_score,
                )

                if request.return_score:
                    # Ensure correct unpacking and type handling
                    if isinstance(formatted_output, tuple) and len(formatted_output) == 2:
                        current_query_docs, current_query_scores = formatted_output
                    else:
                        # logger.error("Type mismatch in batch search formatting with scores. Expected tuple.") # Removed log
                        print(
                            "ERROR: Type mismatch in batch search formatting with scores. Expected tuple."
                        )  # Replaced with print
                        current_query_docs, current_query_scores = [], []  # Fallback
                else:
                    if isinstance(formatted_output, list):
                        current_query_docs = formatted_output
                        current_query_scores = []  # Scores not requested
                    else:
                        # logger.error("Type mismatch in batch search formatting without scores. Expected list.") # Removed log
                        print(
                            "ERROR: Type mismatch in batch search formatting without scores. Expected list."
                        )  # Replaced with print
                        current_query_docs = []  # Fallback
                        current_query_scores = []

            all_query_docs_results.append(current_query_docs)
            if request.return_score:
                all_query_scores_results.append(current_query_scores)

        if request.return_score:
            return all_query_docs_results, all_query_scores_results
        return all_query_docs_results
    except Exception as e:
        # logger.error(f"Error during MOCK batch search: {e}", exc_info=True) # Removed log
        print(f"Error during MOCK batch search: {e}")  # Replaced with print
        raise HTTPException(status_code=500, detail="An error occurred during the mock batch search.")


# --- CLI Handling (Simplified for server only) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Retrieval Server (FlashRAG API)")
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

    # logger.info(f"Starting MOCK retrieval server on http://{args.host}:{args.port}") # Removed log
    print(f"Starting MOCK retrieval server on http://{args.host}:{args.port}")  # Replaced with print
    if args.config:
        # logger.info(f"Note: --config argument ('{args.config}') is ignored by this mock server.") # Removed log
        print(f"Note: --config argument ('{args.config}') is ignored by this mock server.")  # Replaced with print
    try:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        # logger.error("Uvicorn is not installed. Please run: pip install uvicorn[standard]") # Removed log
        print("ERROR: Uvicorn is not installed. Please run: uv install uvicorn[standard]")  # Replaced and suggest uv
    except Exception as e:
        # logger.error(f"Failed to start uvicorn server: {e}", exc_info=True) # Removed log
        print(f"ERROR: Failed to start uvicorn server: {e}")  # Replaced with print

# How to run sections (remains the same)
# ...
