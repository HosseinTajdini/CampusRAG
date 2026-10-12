"""Create and query the dense vector index for CampusRAG."""

import json
import numpy as np

from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
COLLECTION_NAME = "campus_chunks"
MAX_CHARS = 1200
EMBEDDING_BATCH_SIZE = 16
QDRANT_BATCH_SIZE = 64


def split_into_chunks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a document into paragraph-aware chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    paragraphs = [
        paragraph.strip()
        for paragraph in str(text).split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])

            continue

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def _as_document_dicts(documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept either a list of dictionaries or a pandas DataFrame."""
    if hasattr(documents, "to_dict"):
        return documents.to_dict(orient="records")

    return list(documents)


def _json_safe(value: Any) -> Any:
    """Convert metadata into values accepted by Qdrant payloads."""
    return json.loads(
        json.dumps(value, ensure_ascii=False, default=str)
    )


def build_chunk_records(
    documents: Iterable[dict[str, Any]],
    max_chars: int = MAX_CHARS,
) -> list[dict[str, Any]]:
    """Create chunk records and metadata from built documents."""
    records = []

    for document_index, document in enumerate(_as_document_dicts(documents)):
        chunks = document.get("chunks")
        if chunks is None:
            chunks = split_into_chunks(document.get("content", ""), max_chars)

        metadata = {
            key: value
            for key, value in document.items()
            if key not in {"content", "chunks"}
        }
        metadata = _json_safe(metadata)

        for chunk_index, chunk in enumerate(chunks):
            text = str(chunk).strip()
            if not text:
                continue

            records.append({
                "id": str(uuid5(
                    NAMESPACE_URL,
                    f"campus-rag:{document_index}:{chunk_index}",
                )),
                "text": text,
                "metadata": metadata,
            })

    return records


def load_embedding_model(
    model_name: str = MODEL_NAME,
    device: str | None = None,
) -> SentenceTransformer:
    """Load the embedding model used by the project."""
    model_kwargs = {"trust_remote_code": True}
    if device is not None:
        model_kwargs["device"] = device

    return SentenceTransformer(model_name, **model_kwargs)


def add_embeddings(
    records: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode all records and attach their normalized vectors."""
    if not records:
        raise ValueError("No records were created from the documents")

    texts = [record["text"] for record in records]

    vectors = embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    ).astype(np.float32)

    for record, vector in zip(records, vectors):
        record["vector"] = vector.tolist()

    return vectors


def open_qdrant(
    vector_db_path: Path = DEFAULT_VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = 768,
) -> QdrantClient:
    """Open the local Qdrant database and create the collection if needed."""
    vector_db_path = Path(vector_db_path)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(vector_db_path))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    return client


def upsert_records(
    client: QdrantClient,
    records: list[dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = QDRANT_BATCH_SIZE,
) -> None:
    """Upload records and their payloads to Qdrant in batches."""
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        points = [
            models.PointStruct(
                id=record["id"],
                vector=record["vector"],
                payload={
                    "text": record["text"],
                    "metadata": record["metadata"],
                },
            )
            for record in batch
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
        )


def build_vector_database(
    documents: Iterable[dict[str, Any]],
    embedding_model: SentenceTransformer | None = None,
    *,
    model_name: str = MODEL_NAME,
    vector_db_path: Path = DEFAULT_VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    max_chars: int = MAX_CHARS,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    qdrant_batch_size: int = QDRANT_BATCH_SIZE,
    device: str | None = None,
) -> tuple[QdrantClient, list[dict[str, Any]], SentenceTransformer]:
    """Chunk documents, create embeddings, and store them in Qdrant."""
    records = build_chunk_records(documents, max_chars=max_chars)
    print("Number of chunks:", len(records))

    if embedding_model is None:
        embedding_model = load_embedding_model(
            model_name=model_name,
            device=device,
        )

    vectors = add_embeddings(
        records,
        embedding_model,
        batch_size=embedding_batch_size,
    )

    client = open_qdrant(
        vector_db_path=vector_db_path,
        collection_name=collection_name,
        vector_size=vectors.shape[1],
    )

    upsert_records(
        client,
        records,
        collection_name=collection_name,
        batch_size=qdrant_batch_size,
    )

    print("Number of Qdrant points:", client.count(collection_name).count)

    return client, records, embedding_model


def dense_search(
    query: str,
    embedding_model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search Qdrant with a normalized query embedding."""
    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "id": result.id,
            "score": float(result.score),
            "text": result.payload.get("text", ""),
            "metadata": result.payload.get("metadata", {}),
        }
        for result in results
    ]


def load_documents(path: Path = DOCUMENTS_PATH) -> list[dict[str, Any]]:
    """Load the JSONL documents created by build_documents.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Built documents were not found: {path}"
        )

    documents = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    return documents


def main() -> None:
    """Build the vector database from the processed JSONL documents."""
    print("Loading built documents...")
    documents = load_documents()
    print("Number of documents:", len(documents))

    build_vector_database(documents)
    print("Vector database build complete.")


if __name__ == "__main__":
    main()
