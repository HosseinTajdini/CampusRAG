"""Run the CampusRAG data pipeline and start the chat interface."""

import asyncio

from data_processing.build_documents import main as build_documents
from data_processing.rag_pipeline import PROJECT_ROOT, create_gradio_app
from data_processing.vector_store import main as build_vector_database
from data_collector.telegram_collector import main as collect_telegram_data


def main() -> None:
    """Collect data, build the index, and launch Gradio."""
    asyncio.run(collect_telegram_data())
    build_documents()
    build_vector_database()

    app = create_gradio_app()
    app.launch(
        allowed_paths=[
            str(PROJECT_ROOT / "data" / "media"),
        ]
    )


if __name__ == "__main__":
    main()
