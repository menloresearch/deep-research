import argparse
import os
import sys
import typing  # For cast
from pathlib import Path

from openai import OpenAI

from src import search_module  # type: ignore
from src.config import PROCESSED_DATA_DIR as DEFAULT_PROCESSED_DATA_DIR_CONFIG  # type: ignore
from src.config import RAG_SEARCH_RESULTS_COUNT  # type: ignore

# Adjust Python path to import from src
PROJ_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJ_ROOT / "src"
sys.path.insert(0, str(PROJ_ROOT))

# Default configuration
DEFAULT_LLM_MODEL_NAME = "qwen/qwen3-32b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def call_llm(client: OpenAI, prompt: str, model_name: str) -> str:
    """Calls the LLM and returns the response text."""
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = chat_completion.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""


# def perform_search(query: str, bing_key: str, bing_endpoint: str, top_k: int = 3) -> list[dict]: # Old signature
def initialize_local_search(
    cli_processed_data_dir_str: str, config_default_path: Path
) -> bool:
    """Initializes the vector store for local RAG search."""
    if search_module.vectorstore is not None:
        # This case should ideally not be hit if initialize is called once at the start
        print("Local vector store appears to be already loaded.")
        return True

    target_path: Path
    used_default_path = False

    # Determine the correct path for PROCESSED_DATA_DIR
    # Compare string form of default path from config with CLI arg to see if user specified a different one.
    if cli_processed_data_dir_str != str(config_default_path):
        target_path = Path(cli_processed_data_dir_str)
        print(f"Using custom processed data directory from CLI argument: {target_path}")
    else:
        target_path = config_default_path
        used_default_path = True
        print(f"Using default processed data directory from config: {target_path}")

    # Validate the chosen path
    if not target_path.exists() or not (target_path / "index.faiss").exists():
        path_type_msg = "default" if used_default_path else "specified"
        print(
            f"Error: Processed data directory '{target_path}' (from {path_type_msg} path) "
            f"does not exist or 'index.faiss' not found inside."
        )
        print("Please ensure a valid FAISS index exists at the location.")
        return False

    # Set the PROCESSED_DATA_DIR in the search_module (as it uses this global)
    search_module.PROCESSED_DATA_DIR = target_path

    print(f"\nLoading local vector store from: {search_module.PROCESSED_DATA_DIR}...")
    search_module.vectorstore = search_module.load_vectorstore()

    if search_module.vectorstore is None:
        print(
            f"Failed to load vector store from {search_module.PROCESSED_DATA_DIR}. Exiting."
        )
        return False

    print("Local vector store loaded successfully.")
    return True


def perform_local_search(
    query: str, top_k: int = RAG_SEARCH_RESULTS_COUNT
) -> list[str]:
    """Performs a local RAG search and returns top_k document contents as a list of strings."""
    if search_module.vectorstore is None:
        print(
            "Error: Vectorstore not loaded. Ensure it's initialized before searching."
        )
        return []
    try:
        results_from_search_module = search_module.search(query, results=top_k)

        if isinstance(results_from_search_module, list):
            if all(isinstance(item, str) for item in results_from_search_module):
                return typing.cast(
                    list[str], results_from_search_module
                )  # Cast to list[str] for clarity
            else:
                print("Error: Local search returned a list with non-string elements.")
                return []
        else:
            print(
                f"Error: Local search did not return a list as expected. Got type: {type(results_from_search_module)}"
            )
            return []
    except Exception as e:
        print(
            f"An unexpected error occurred during local search for query '{query}': {e}"
        )
        return []


# def format_search_results_for_llm(results: list[dict]) -> str: # Old signature
def format_search_results_for_llm(documents: list[str]) -> str:
    """Formats local RAG search results (list of document strings) for the LLM prompt."""
    if not documents:
        return "No search results found."

    formatted = "Retrieved Documents:\n"
    for i, doc_content in enumerate(documents):
        formatted += f"Document {i + 1}:\n"
        # formatted += f"  Title: {res['title']}\n" # No longer applicable
        # formatted += f"  URL: {res['url']}\n" # No longer applicable
        # formatted += f"  Snippet: {res['snippet']}\n\n" # No longer applicable
        formatted += f"{doc_content}\n\n"  # Use the full document content
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple CLI Research Agent with Local RAG"
    )
    parser.add_argument("question", type=str, help="The research question to answer.")
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API Key. Defaults to OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DATA_DIR_CONFIG),
        help=f"Path to the processed data directory containing the FAISS index. Defaults to {DEFAULT_PROCESSED_DATA_DIR_CONFIG}",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL_NAME,
        help=f"LLM model to use via OpenRouter. Defaults to {DEFAULT_LLM_MODEL_NAME}",
    )
    parser.add_argument(
        "--top_k_search",
        type=int,
        default=RAG_SEARCH_RESULTS_COUNT,
        help=f"Number of search results to retrieve and use. Defaults to {RAG_SEARCH_RESULTS_COUNT} (from config).",
    )

    args = parser.parse_args()

    if not args.openrouter_api_key:
        print(
            "Error: OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or use --openrouter_api_key argument."
        )
        return

    # Initialize local RAG search capabilities
    if not initialize_local_search(
        args.processed_data_dir, DEFAULT_PROCESSED_DATA_DIR_CONFIG
    ):
        return  # Exit if RAG initialization failed

    print(f'Answering question: "{args.question}"')
    print(f"Using LLM: {args.llm_model}")
    # print(f"Using Bing endpoint: {args.bing_endpoint}") # Removed
    print(f"Retrieving top {args.top_k_search} local RAG search results.")

    try:
        llm_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=args.openrouter_api_key,
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    # Step 1: Generate Search Query (still useful to refine question for RAG)
    print("\nStep 1: Generating search query (for local RAG)...")
    query_generation_prompt = (
        f'Given the research question: "{args.question}", '
        "generate a concise and effective search query to find information within a local document collection. "  # Modified for RAG context
        "Respond with ONLY the search query itself, and nothing else."
    )
    search_query = call_llm(llm_client, query_generation_prompt, args.llm_model)

    if not search_query:
        print(
            "Failed to generate search query for local RAG. Using original question as query."
        )
        search_query = args.question  # Fallback to original question
    print(f"Generated RAG Search Query: {search_query}")

    # Step 2: Perform Local RAG Search
    print("\nStep 2: Performing local RAG search...")
    # search_results = perform_search(search_query, args.bing_api_key, args.bing_endpoint, args.top_k_search) # Old call
    retrieved_documents = perform_local_search(search_query, args.top_k_search)

    if not retrieved_documents:
        print(
            "No documents retrieved from local RAG or an error occurred during search. Exiting."
        )
        return

    formatted_results_str = format_search_results_for_llm(retrieved_documents)
    print(f"Retrieved Documents (Processed for LLM):\n{formatted_results_str}")

    # Step 3: Generate Final Answer
    print("\nStep 3: Generating final answer based on local RAG results...")
    answer_generation_prompt = (
        f'Question: "{args.question}"\n\n'
        "Based on the following retrieved documents from a local knowledge base, please provide a concise answer to the question. "  # Modified for RAG context
        "Cite the relevant document numbers (e.g., [Document 1], [Document 2]) if applicable. "
        "Do not make up information not present in the retrieved documents.\n\n"
        f"{formatted_results_str}\n\n"
        "Your Answer:"
    )
    final_answer = call_llm(llm_client, answer_generation_prompt, args.llm_model)

    if not final_answer:
        print("Failed to generate final answer. Exiting.")
        return

    print("\n--- Final Answer ---")
    print(final_answer)
    print("--------------------")


if __name__ == "__main__":
    main()
