"""Script to build the FAISS index from the knowledge base articles.

Run from the project root:
    python -m src.data.seed_articles
"""
import logging
import sys
from pathlib import Path

# Allow running from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.indexer import FAISSIndexer
from src.utils.logging_config import setup_logging

logger = setup_logging("INFO", "finnie.seed")


def main() -> None:
    articles_dir = PROJECT_ROOT / "src" / "data" / "articles"
    if not articles_dir.exists():
        logger.error("Articles directory not found: %s", articles_dir)
        sys.exit(1)

    md_files = list(articles_dir.rglob("*.md"))
    logger.info("Found %d markdown articles in %s", len(md_files), articles_dir)

    if not md_files:
        logger.error("No markdown files found. Add articles first.")
        sys.exit(1)

    logger.info("Building FAISS index… (this may take a minute on first run)")
    indexer = FAISSIndexer()

    try:
        indexer.build_from_directory(articles_dir)
        logger.info(
            "Index built successfully: %d chunks from %d articles",
            len(indexer.chunks),
            len(md_files),
        )
    except Exception as exc:
        logger.error("Failed to build index: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
