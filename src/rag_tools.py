import os
import re
import json
import math
import logging
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from pypdf import PdfReader
from pydantic import BaseModel, ValidationError

try:
    import faiss
except Exception:
    faiss = None

try:
    import openai
except Exception:
    openai = None

LOG_DIR = Path(__file__).parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_DIR / "weather.log", level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def extract_text_from_pdf(path: str) -> str:
    p = Path(path)
    reader = PdfReader(str(p))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def _simple_embedding(texts: List[str], dim: int = 512) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype='float32')
    for i, t in enumerate(texts):
        tokens = re.findall(r"\w+", t.lower())
        for tok in tokens:
            idx = (abs(hash(tok)) % dim)
            vectors[i, idx] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


def get_embeddings(texts: List[str], model: str = "openai") -> np.ndarray:
    """
    Use OpenAI embeddings if OPENAI_API_KEY is set and openai package is present.
    Supports both old and new openai-python interfaces. Falls back to local embedding on failure.
    """
    if os.getenv("OPENAI_API_KEY") and openai is not None:
        try:
            # New API: client-based
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
                embs = []
                for d in resp.data:
                    if hasattr(d, "embedding"):
                        vec = d.embedding
                    elif isinstance(d, dict) and "embedding" in d:
                        vec = d["embedding"]
                    else:
                        raise ValueError("Unexpected embedding response format")
                    embs.append(np.array(vec, dtype='float32'))
                return np.vstack(embs)
            # Fallback for older interface (if installed)
            elif hasattr(openai, "Embedding") and hasattr(openai.Embedding, "create"):
                resp = openai.Embedding.create(input=texts, model="text-embedding-3-small")
                embs = [np.array(r["embedding"], dtype='float32') for r in resp["data"]]
                return np.vstack(embs)
        except Exception as e:
            logging.warning("OpenAI embeddings failed, falling back to local embedding: %s", e)
            return _simple_embedding(texts)
    return _simple_embedding(texts)


def build_faiss_index(vectors: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss not available in this environment")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim) if vectors.dtype == np.float32 else faiss.IndexFlatL2(dim)
    if isinstance(index, faiss.IndexFlatIP):
        faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def search_index(index, vectors: np.ndarray, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    q = query_vector.reshape(1, -1).astype('float32')
    if hasattr(faiss, 'normalize_L2'):
        faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    results = [(int(idx), float(dist)) for idx, dist in zip(I[0], D[0])]
    return results


class WeatherRequest(BaseModel):
    fecha: str


def get_weather(fecha: str) -> dict:
    try:
        req = WeatherRequest(fecha=fecha)
    except ValidationError as e:
        logging.error(f"get_weather validation error: {e}")
        raise
    seed = abs(hash(fecha)) % 100
    temp = 10 + (seed % 20)
    result = {"fecha": fecha, "temperature_c": temp, "condition": "sunny" if seed % 3 == 0 else "cloudy"}
    logging.info(json.dumps({"request": req.dict(), "result": result}))
    return result


class ConversationMemory:
    def __init__(self):
        self.history = []

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_system(self, text: str):
        self.history.append({"role": "system", "content": text})

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def get_context(self, max_tokens: int = 1500) -> List[dict]:
        out = []
        tokens = 0
        for msg in reversed(self.history):
            tokens += len(msg["content"]) // 4
            if tokens > max_tokens:
                break
            out.append(msg)
        return list(reversed(out))


def approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def save_vectors_and_chunks(vectors: np.ndarray, chunks: List[str], out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'vectors.npy', vectors)
    with open(out / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out


def save_faiss_index(index, path: str):
    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if faiss is None:
        raise RuntimeError('faiss not available')
    faiss.write_index(index, str(outp))
    return outp


def load_faiss_index(path: str, dim: int):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if faiss is None:
        raise RuntimeError('faiss not available')
    index = faiss.read_index(str(p))
    return index