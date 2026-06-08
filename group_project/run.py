"""One-command launcher for the group RAG chatbot."""

import sys

from src.run_app import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
