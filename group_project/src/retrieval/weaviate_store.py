"""Weaviate Local vector store integration."""

from __future__ import annotations

import json
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

from src.chunking.legal_chunker import chunk_documents
from src.config import CONFIG
from src.retrieval.embeddings import embed_query, embed_texts


def connect_local():
    """Connect to Weaviate local Docker instance."""
    return weaviate.connect_to_local(
        host=CONFIG.weaviate_host,
        port=CONFIG.weaviate_http_port,
        grpc_port=CONFIG.weaviate_grpc_port,
    )


def ensure_collection(client) -> Any:
    """Create collection with externally supplied OpenAI vectors."""
    if client.collections.exists(CONFIG.weaviate_collection):
        return client.collections.get(CONFIG.weaviate_collection)

    return client.collections.create(
        name=CONFIG.weaviate_collection,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="source_path", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="metadata_json", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
        ],
    )


def reset_collection(client) -> None:
    if client.collections.exists(CONFIG.weaviate_collection):
        client.collections.delete(CONFIG.weaviate_collection)
    ensure_collection(client)


def index_chunks_to_weaviate(chunks: list[dict[str, Any]] | None = None, reset: bool = False) -> int:
    """
    Index all chunks into Weaviate local.

    Requires Docker service from group_project/docker-compose.yml.
    """
    selected_chunks = chunks if chunks is not None else chunk_documents()
    client = connect_local()
    try:
        if reset:
            reset_collection(client)
        collection = ensure_collection(client)

        vectors = embed_texts([chunk["content"] for chunk in selected_chunks], batch_size=32)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(selected_chunks, vectors, strict=True):
                metadata = chunk.get("metadata", {})
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": str(metadata.get("source", "")),
                        "source_path": str(metadata.get("source_path", "")),
                        "doc_type": str(metadata.get("doc_type", "")),
                        "title": str(metadata.get("title", "")),
                        "metadata_json": json.dumps(metadata, ensure_ascii=False),
                        "chunk_id": chunk["id"],
                    },
                    vector=vector,
                )
        return len(selected_chunks)
    finally:
        client.close()


def weaviate_dense_search(query: str, top_k: int = 10, query_vector: list[float] | None = None) -> list[dict[str, Any]]:
    """Dense retrieval from Weaviate using near_vector."""
    client = connect_local()
    try:
        collection = ensure_collection(client)
        vector = query_vector if query_vector is not None else embed_query(query)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results: list[dict[str, Any]] = []
        for rank, obj in enumerate(response.objects, start=1):
            props = obj.properties
            metadata = json.loads(props.get("metadata_json") or "{}")
            distance = float(obj.metadata.distance or 0.0)
            score = max(0.0, 1.0 - distance)
            metadata.update({"retriever": "weaviate_dense", "rank": rank, "chunk_id": props.get("chunk_id")})
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": score,
                    "metadata": metadata,
                    "source": "weaviate_dense",
                }
            )
        return results
    finally:
        client.close()
