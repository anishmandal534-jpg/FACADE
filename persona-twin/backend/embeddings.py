import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Load the PROJECT ROOT .env explicitly
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ROOT_ENV = PROJECT_ROOT / ".env"
BACKEND_ENV = BASE_DIR / ".env"

# Root .env is the primary configuration.
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV, override=True)

# Backend .env is only a fallback for variables not already set.
if BACKEND_ENV.exists():
    load_dotenv(BACKEND_ENV, override=False)


# ---------------------------------------------------------
# OpenRouter configuration
# ---------------------------------------------------------

OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY", "").strip()
    or os.getenv("DEEPSEEK_API_KEY", "").strip()
)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
).strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
).strip()

# Qdrant collection uses 384-dimensional vectors.
EMBEDDING_DIMENSIONS = 384


# ---------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------

_client = None

if OPENROUTER_API_KEY:
    _client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL",
                "http://127.0.0.1:8000"
            ),
            "X-Title": os.getenv(
                "OPENROUTER_APP_NAME",
                "FACADE Persona Twin"
            ),
        },
    )


# ---------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------

def embed_texts(texts):
    """
    Generate embeddings through OpenRouter.

    The returned vectors remain in the same order as the
    supplied input texts.
    """

    if _client is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    if isinstance(texts, str):
        texts = [texts]

    texts = [str(text) for text in texts]

    if not texts:
        return []

    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )

    ordered = sorted(
        response.data,
        key=lambda item: item.index
    )

    return [
        item.embedding
        for item in ordered
    ]


def embed_text(text):
    """
    Generate one embedding vector.
    """

    vectors = embed_texts([text])

    return vectors[0]