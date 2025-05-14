import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import RAG_SEARCH_RESULTS_COUNT, logger

# Relative imports from the parent directory (src/)
from ..knowledge_base import _vectorstore, load_vectorstore, simple_rag_search

# Configuration for the server
HOST = "0.0.0.0"
PORT = int(os.getenv("SIMPLE_RETRIEVAL_PORT", "8001"))  # Use ENV var for port

app = FastAPI(
    title="Simple Retrieval Server",
    description="A simple server for FAISS-based semantic search.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    k: int | None = Field(default=RAG_SEARCH_RESULTS_COUNT, gt=0, le=100)


class SearchResultItem(BaseModel):
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


@app.on_event("startup")
async def startup_event():
    logger.info("Starting server and loading vectorstore...")
    load_vectorstore()  # Attempt to load the vectorstore
    if _vectorstore is None:
        logger.error("Vectorstore could not be loaded. Search functionality will be unavailable.")
    else:
        logger.info("Vectorstore loaded successfully.")


@app.get("/health", summary="Check Health", description="Returns a 200 OK if the server is running.")
async def health_check():
    return {"status": "ok"}


@app.post(
    "/retrieve",
    response_model=SearchResponse,
    summary="Perform Search",
    description="Searches the vectorstore for relevant documents.",
)
async def search(request: SearchRequest):
    if _vectorstore is None:
        logger.error("Search attempted but vectorstore is not loaded.")
        raise HTTPException(status_code=503, detail="Vectorstore not available. Please try again later.")

    logger.info(f"Received search query: '{request.query}' with k={request.k}")
    try:
        search_results = simple_rag_search(query=request.query, k=request.k or RAG_SEARCH_RESULTS_COUNT)
        # Ensure results are in the correct format for SearchResponse
        formatted_results = [
            SearchResultItem(content=item.get("content", ""), metadata=item.get("metadata", {}))
            for item in search_results
        ]
        return SearchResponse(results=formatted_results)
    except Exception as e:
        logger.error(f"Error during search for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during the search.")


# To run this server, use the Makefile command: make run-simple-retrieval-server
# Or directly with uvicorn: PYTHONPATH=. uvicorn src.search.retrieval_server:app --host 0.0.0.0 --port 8001 (or your configured port)
