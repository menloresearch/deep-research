# src/embeddings.py
import torch
from langchain.embeddings.base import Embeddings  # If you use Langchain's base

# from langchain_core.embeddings import Embeddings # Newer Langchain path
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_EMBEDDING_MODEL, logger  # Use your central logger

# Set a default model here
# DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Removed

print(f"[DEBUG] EMBEDDINGS INIT: Module imported, DEFAULT_EMBEDDING_MODEL={DEFAULT_EMBEDDING_MODEL}")


class CustomHuggingFaceEmbeddings(Embeddings):
    """
    A custom embeddings class that wraps a Hugging Face model for generating embeddings.
    Supports two modes:
    - "sentence": uses the [CLS] token representation for sentence/document embeddings.
    - "query": uses mean pooling over tokens (weighted by the attention mask) for query embeddings.
    """

    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, default_mode="sentence"):
        print(f"[DEBUG] EMBEDDINGS: Initializing with model_name={model_name}, default_mode={default_mode}")
        self.model_name = model_name
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] EMBEDDINGS: Using device: {self.device}")
            logger.info(f"Using device: {self.device} for embeddings model {self.model_name}")

            print(f"[DEBUG] EMBEDDINGS: Loading model {model_name}...")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            print(f"[DEBUG] EMBEDDINGS: Model loaded successfully, type={type(self.model)}")

            print(f"[DEBUG] EMBEDDINGS: Loading tokenizer {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"[DEBUG] EMBEDDINGS: Tokenizer loaded successfully, type={type(self.tokenizer)}")

            self.default_mode = default_mode  # "sentence" or "query"
            self.model.eval()  # Set model to evaluation mode
            print("[DEBUG] EMBEDDINGS: Initialization complete")
        except Exception as e:
            print(f"[ERROR] EMBEDDINGS: Failed to load embedding model '{model_name}': {e}")
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            import traceback

            traceback.print_exc()
            raise  # Re-raise the exception to halt if model loading fails

    def _get_embedding_vectors(self, texts: list[str], mode: str) -> torch.Tensor:
        """Internal method to get embedding vectors."""
        print(f"[DEBUG] EMBEDDINGS: Getting embeddings for {len(texts)} texts in mode '{mode}'")
        if not texts:
            print("[DEBUG] EMBEDDINGS: Empty texts list, returning empty tensor")
            return torch.empty(0, self.model.config.hidden_size).to(self.device)

        assert mode in (
            "query",
            "sentence",
        ), f"Unsupported mode: {mode}. Only 'query' and 'sentence' are supported."

        # Tokenize the input texts
        print("[DEBUG] EMBEDDINGS: Tokenizing texts")
        inp = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"[DEBUG] EMBEDDINGS: Tokenization complete, input shape: {inp['input_ids'].shape}")
        inp = {key: value.to(self.device) for key, value in inp.items()}

        with torch.no_grad():
            print("[DEBUG] EMBEDDINGS: Running model forward pass")
            output = self.model(**inp)
            print("[DEBUG] EMBEDDINGS: Model forward pass complete")

        if mode == "query":
            # Mean pooling: weight by attention mask and average across tokens
            print("[DEBUG] EMBEDDINGS: Using query mode (mean pooling)")
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:  # sentence mode
            print("[DEBUG] EMBEDDINGS: Using sentence mode (CLS token)")
            vectors = output.last_hidden_state[:, 0, :]  # CLS token

        print(f"[DEBUG] EMBEDDINGS: Embeddings generated, shape: {vectors.shape}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of documents (using sentence mode)."""
        print(f"[DEBUG] EMBEDDINGS: embed_documents called with {len(texts)} texts")
        if not texts:
            print("[DEBUG] EMBEDDINGS: Empty texts list, returning empty list")
            return []
        vectors = self._get_embedding_vectors(texts, mode="sentence")
        result = vectors.cpu().numpy().tolist()
        print(f"[DEBUG] EMBEDDINGS: Returning {len(result)} document embeddings")
        return result

    def embed_query(self, text: str) -> list[float]:
        """Compute an embedding for a single query (using query mode)."""
        print(f"[DEBUG] EMBEDDINGS: embed_query called with text: '{text[:30]}...' (truncated)")
        if not text:  # Handle empty query string
            # Return a zero vector or raise an error, depending on desired behavior
            logger.warning("Embedding empty query string.")
            print("[DEBUG] EMBEDDINGS: Empty query text, returning zero vector")
            # Assuming model hidden size can be accessed, otherwise define a fixed size
            hidden_size = self.model.config.hidden_size if hasattr(self.model, "config") else 768
            return [0.0] * hidden_size

        vectors = self._get_embedding_vectors([text], mode="query")
        result = vectors.cpu().numpy()[0].tolist()
        print(f"[DEBUG] EMBEDDINGS: Returning query embedding of size {len(result)}")
        return result


# For quick testing
if __name__ == "__main__":
    logger.info("Testing CustomHuggingFaceEmbeddings...")
    try:
        embeddings = CustomHuggingFaceEmbeddings()
        texts = [
            "Illustration of the REaLTabFormer model.",
            "Predicting human mobility holds significant practical value.",
            "Workforce preparation for emerging labor demands.",
        ]
        doc_embeddings = embeddings.embed_documents(texts)
        logger.info(
            f"Document embeddings shape: ({len(doc_embeddings)}, {len(doc_embeddings[0]) if doc_embeddings else 0})"
        )

        query_embedding = embeddings.embed_query("Which sentence talks about jobs?")
        logger.info(f"Query embedding shape: ({len(query_embedding)})")
        logger.info("Embeddings test successful.")
    except Exception as e:
        logger.error(f"Embeddings test failed: {e}")
