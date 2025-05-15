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
    print("Mock load_vectorstore called. Setting mock vectorstore.")
    _vectorstore = "MockVectorStoreInstanceLoaded"


def _initial_mock_simple_rag_search(query: str, k: int) -> list[dict]:
    if _vectorstore is None:
        print("ERROR: Mock search called but mock vectorstore is not loaded.")
        return []
    results = []
    for i in range(k):
        results.append(
            {
                "content": f"Mock search result {i + 1} for query: {query}",
                "metadata": {
                    "doc_id": f"mock_doc_id_{i + 1}",
                    "source": "mock_source_data",
                    "score": 0.95 - (i * 0.05),  # Mock score
                },
            }
        )
    return results[:k]


# Initialize with fallbacks. These will be overwritten if real imports succeed.
load_vectorstore = _initial_mock_load_vectorstore
simple_rag_search = _initial_mock_simple_rag_search

try:
    from ...src.config import RAG_SEARCH_RESULTS_COUNT as _real_rag_count
    from ...src.knowledge_base import _vectorstore as _kb_vectorstore
    from ...src.knowledge_base import load_vectorstore as _real_load_vectorstore
    from ...src.knowledge_base import simple_rag_search as _real_simple_rag_search

    _vectorstore = _kb_vectorstore
    load_vectorstore = _real_load_vectorstore
    simple_rag_search = _real_simple_rag_search
    RAG_SEARCH_RESULTS_COUNT = _real_rag_count
    print("Successfully imported real components.")

except ImportError:
    print(
        "Warning: Running with standalone script. Relative imports failed. Using mock components."
    )
    print(
        "Ensure PYTHONPATH is set correctly (e.g., to project root) if expecting real components."
    )
    # Fallbacks are already set.

# Configuration for the server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("SIMPLE_RETRIEVER_PORT", "8002"))

app = FastAPI(
    title="Simple Retrieval Server (FlashRAG API Mock)",
    description="A simplified server mimicking FlashRAG retriever API with mock data.",
    version="0.4.0",  # Incremented version
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
    print("Server starting up. Loading mock vectorstore...")
    load_vectorstore()
    if _vectorstore:
        print("Mock vectorstore loaded successfully.")
    else:
        print("ERROR: Mock vectorstore failed to load.")


@app.get("/health")
async def health_check() -> dict:  # Changed from dict[str, any] to dict
    available = 1 if _vectorstore is not None else 0
    total = 1  # This simple server always conceptually has one retriever
    status = "healthy" if _vectorstore is not None else "unhealthy"
    return {
        "status": status,
        "retrievers": {
            "total": total,
            "available": available,
        },
    }


def _perform_internal_search(
    query_text: str, top_n_results: int
) -> list[dict]:  # list[dict] here too
    return simple_rag_search(query=query_text, k=top_n_results)


def _format_results_for_api(
    search_results_list: list[dict],  # list[dict] here too
    return_score_flag: bool,
) -> tuple[list[Document], list[float]] | list[Document]:
    docs_list = []
    scores_list = []
    for i, item_dict in enumerate(search_results_list):
        doc_id_str = str(
            item_dict.get("metadata", {}).get("doc_id", f"fallback_doc_id_{i}")
        )
        content_str = str(item_dict.get("content", ""))
        docs_list.append(Document(id=doc_id_str, contents=content_str))
        if return_score_flag:
            score_val = float(item_dict.get("metadata", {}).get("score", 1.0))
            scores_list.append(score_val)

    if return_score_flag:
        return docs_list, scores_list
    return docs_list


@app.post("/search")
async def search_endpoint(
    request: QueryRequest,
) -> tuple[list[Document], list[float]] | list[Document]:
    if _vectorstore is None:
        print("Search request failed: Vectorstore not available.")
        raise HTTPException(
            status_code=503, detail="Vectorstore not available. Please try again later."
        )
    if not request.query or not request.query.strip():
        print("Search request with empty query.")
        raise HTTPException(status_code=400, detail="Query content cannot be empty.")

    print(
        f"Processing search: '{request.query}', top_n={request.top_n}, return_score={request.return_score}"
    )
    try:
        raw_search_results = _perform_internal_search(
            query_text=request.query, top_n_results=request.top_n
        )
        return _format_results_for_api(
            search_results_list=raw_search_results,
            return_score_flag=request.return_score,
        )
    except Exception as e:
        print(f"Error during search for query '{request.query}': {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during the search."
        )


@app.post("/batch_search")
async def batch_search_endpoint(
    request: BatchQueryRequest,
) -> tuple[list[list[Document]], list[list[float]]] | list[list[Document]]:
    if _vectorstore is None:
        print("Batch search request failed: Vectorstore not available.")
        raise HTTPException(
            status_code=503, detail="Vectorstore not available. Please try again later."
        )
    if not request.query:
        print("Batch search request with empty query list.")
        raise HTTPException(
            status_code=400, detail="Query list cannot be empty for batch search."
        )

    print(
        f"Processing batch search for {len(request.query)} queries, top_n={request.top_n}, return_score={request.return_score}"
    )
    all_query_docs_results: list[list[Document]] = []
    all_query_scores_results: list[list[float]] = []

    try:
        for single_query_text in request.query:
            current_query_docs: list[Document] = []
            current_query_scores: list[float] = []
            if not single_query_text or not single_query_text.strip():
                print(
                    f"Warning: Empty query string in batch: '{single_query_text}'. Returning empty results for this query."
                )
            else:
                raw_search_results = _perform_internal_search(
                    query_text=single_query_text, top_n_results=request.top_n
                )
                formatted_output = _format_results_for_api(
                    search_results_list=raw_search_results,
                    return_score_flag=request.return_score,
                )
                if request.return_score:
                    docs, scores = (
                        formatted_output  # Re-add type: ignore if necessary, but check if unpacking itself is an issue for linter
                    )
                    current_query_docs = docs  # type: ignore
                    current_query_scores = scores  # type: ignore
                else:
                    current_query_docs = formatted_output  # type: ignore
            all_query_docs_results.append(current_query_docs)
            if request.return_score:
                all_query_scores_results.append(current_query_scores)
        if request.return_score:
            return all_query_docs_results, all_query_scores_results
        return all_query_docs_results
    except Exception as e:
        print(f"Error during batch search: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during batch search."
        )


# --- CLI Handling (Simplified for server only) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Retrieval Server (FlashRAG API Mock)"
    )
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
    parser.add_argument(
        "--host", type=str, default=HOST, help=f"Host (default: {HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Port (default: {PORT})"
    )
    args = parser.parse_args()

    print(f"Starting Simple retrieval server on http://{args.host}:{args.port}")
    if args.config:
        print(
            f"Note: --config argument ('{args.config}') is ignored by this mock server."
        )

    try:
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print(
            "ERROR: Uvicorn is not installed. Please run: uv install uvicorn[standard]"
        )
    except Exception as e:
        print(f"ERROR: Failed to start uvicorn server: {e}")
