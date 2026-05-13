"""Entry point for the Finnie AI Finance Assistant Streamlit app."""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
INDEX_DIR = PROJECT_ROOT / "src" / "data" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"


def _build_index_if_missing() -> None:
    """Build the FAISS knowledge base index on first run (e.g. HuggingFace Spaces)."""
    if INDEX_FILE.exists():
        return
    print("FAISS index not found — building knowledge base index (this takes ~2 minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "src.data.seed_articles"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("Warning: index build failed. RAG retrieval will be unavailable.")


if __name__ == "__main__":
    _build_index_if_missing()
    app_path = PROJECT_ROOT / "src" / "web_app" / "app.py"
    sys.exit(
        subprocess.call(
            ["streamlit", "run", str(app_path), "--server.headless", "true"],
            cwd=str(PROJECT_ROOT),
        )
    )
