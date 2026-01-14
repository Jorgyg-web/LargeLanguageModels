#!/usr/bin/env python3
from pathlib import Path
import json
import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path so `src` is importable when running scripts directly
repo_root = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.rag_tools import extract_text_from_pdf, chunk_text, get_embeddings, save_vectors_and_chunks, respond_with_rag, build_faiss_index
import numpy as np

DATA = Path('data') / 'TENERIFE.pdf'
ART = Path('artifacts') / 'tenerife_index'
ART.mkdir(parents=True, exist_ok=True)

print("Extracting PDF...")
text = extract_text_from_pdf(str(DATA))
chunks = chunk_text(text, chunk_size=500, overlap=100)
print(f"Chunks: {len(chunks)}")

print("Computing embeddings...")
vectors = get_embeddings(chunks)
np.save(ART / 'vectors.npy', vectors)
with open(ART / 'chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# try faiss
try:
    index = build_faiss_index(vectors)
    print("FAISS index built")
except Exception as e:
    index = None
    print("FAISS not available:", e)

print("Running RAG + LLM sample query...")
res = respond_with_rag("¿Dónde están las mejores playas para familias en Tenerife?", vectors=vectors, chunks=chunks, index=index, top_k=3, use_llm=True)
print("Answer:", res["answer"][:400])
print("Sources:", res["sources"])
with open(ART / 'full_pipeline_result.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("Saved result to", ART / 'full_pipeline_result.json')
