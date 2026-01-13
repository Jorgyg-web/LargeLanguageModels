#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Make repo root importable so `src` can be found both locally and in CI
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    from src.rag_tools import get_embeddings
except Exception as e:
    print("IMPORT ERROR:", e)
    raise SystemExit(2)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Configure repository secret 'OPENAI_API_KEY'.")
        raise SystemExit(1)
    try:
        embs = get_embeddings(["smoke test"])
        if embs is None:
            print("SMOKE FAIL: get_embeddings returned None")
            raise SystemExit(2)
        print("OK: embeddings shape:", embs.shape)
        # Non-sensitive numeric summary for sanity check
        print("emb_sum:", float(embs[0].sum()))
    except Exception as e:
        print("SMOKE FAIL:", e)
        raise SystemExit(2)

if __name__ == "__main__":
    main()