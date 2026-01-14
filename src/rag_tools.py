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


# ---------- RAG + LLM responder con citas + soporte function-calling ----------
def _compose_prompt(query: str, top_chunks: list) -> str:
    ctx = "\n\n---\n\n".join(f"Source {i+1}:\n{c[:800]}" for i, c in enumerate(top_chunks))
    prompt = (
        "Eres un asistente turístico. Usa la información de las fuentes abajo para responder.\n\n"
        f"{ctx}\n\n"
        f"Pregunta: {query}\n\n"
        "Devuelve una respuesta concisa y cita las fuentes (Source 1, Source 2...)."
    )
    return prompt


def respond_with_rag(query: str, vectors=None, chunks=None, index=None, top_k: int = 3, use_llm: bool = True):
    """
    Retrieve top-k chunks for `query`, compose a prompt and optionally call the LLM.
    Returns dict with `answer`, `sources` and `chunks` used.
    """
    if vectors is None or chunks is None:
        raise ValueError("vectors and chunks must be provided")

    # compute query vector and do brute-force cosine retrieval
    qv = get_embeddings([query])[0]
    sims = []
    for v in vectors:
        a = qv / (np.linalg.norm(qv) + 1e-12)
        b = v / (np.linalg.norm(v) + 1e-12)
        sims.append(float(np.dot(a, b)))
    top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    top_chunks = [chunks[i] for i in top_idx]
    prompt = _compose_prompt(query, top_chunks)

    # Call LLM if requested and available
    if use_llm and os.getenv("OPENAI_API_KEY") and openai is not None:
        try:
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                # Try the responses/generative API if available
                if hasattr(client, "responses"):
                    resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=512)
                    # extract text from response object/dict
                    if hasattr(resp, "output_text") and resp.output_text:
                        text = resp.output_text
                    else:
                        # fallback extraction
                        text = ""
                        if isinstance(resp, dict):
                            text = resp.get("output_text") or resp.get("text") or str(resp)
                    return {"answer": text.strip(), "sources": [f"Source {i+1}" for i in range(len(top_chunks))], "chunks": top_chunks}
            # older ChatCompletion fallback
            if hasattr(openai, "ChatCompletion"):
                msg = [{"role": "user", "content": prompt}]
                resp = openai.ChatCompletion.create(model="gpt-4o", messages=msg, max_tokens=512)
                text = resp["choices"][0]["message"]["content"]
                return {"answer": text.strip(), "sources": [f"Source {i+1}" for i in range(len(top_chunks))], "chunks": top_chunks}
        except Exception as e:
            logging.warning("LLM call failed, returning retrieval-only summary: %s", e)

    # Fallback: return retrieval context as simple answer
    summary = prompt[:1200]
    return {"answer": summary, "sources": [f"Source {i+1}" for i in range(len(top_chunks))], "chunks": top_chunks}

