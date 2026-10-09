import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

IMPERSONATED_NAME = os.getenv("IMPERSONATED_NAME", "Taylor").strip() or "Taylor"

# === Model settings ===
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# === Retrieval and conversation settings ===
SESSION_GAP_MINUTES = int(os.getenv("SESSION_GAP_MINUTES", "180"))
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "4"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "5"))
MAX_RANDOM_CHECKS = int(os.getenv("MAX_RANDOM_CHECKS", "3"))
RANDOM_SESSIONS_PER_CHECK = int(os.getenv("RANDOM_SESSIONS_PER_CHECK", "3"))
