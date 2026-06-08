"""Build Weaviate and Whoosh indexes for the group chatbot."""

from __future__ import annotations

import argparse

from src.run_app import build_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index group RAG data")
    parser.add_argument("--skip-weaviate", action="store_true", help="Only build local Whoosh BM25")
    parser.add_argument("--skip-whoosh", action="store_true", help="Only build Weaviate dense index")
    parser.add_argument("--reset", action="store_true", help="Reset indexes before writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_whoosh:
        raise ValueError("--skip-whoosh is deprecated in this app; Whoosh is the local fallback index.")
    build_indexes(skip_weaviate=args.skip_weaviate, reset=args.reset)


if __name__ == "__main__":
    main()
