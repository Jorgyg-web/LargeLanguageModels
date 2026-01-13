#!/usr/bin/env python3
import os
import sys
from src.rag_tools import get_embeddings

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Configure repository secret 'OPENAI_API_KEY'.")
        raise SystemExit(1)
    try:
        embs = get_embeddings(["smoke test"])
        print("OK: embeddings shape:", embs.shape)
        print("emb_sum:", float(embs[0].sum()))
    except Exception as e:
        print("SMOKE FAIL:", e)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
