"""Run data collection, document building, and dense vector indexing."""

import asyncio

from data_processing.build_documents import main as build_documents
from data_processing.vector_store import main as build_vector_database
from data_collector.telegram_collector import main as collect_telegram_data


def main() -> None:
    """Collect data, build documents, and index them in Qdrant."""
    asyncio.run(collect_telegram_data())
    build_documents()
    build_vector_database()


if __name__ == "__main__":
    main()
