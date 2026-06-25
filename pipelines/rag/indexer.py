# pipelines/rag/indexer.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
import uuid

from pipelines.models import Chunk
from pipelines.rag.embeddings import get_dense_embedder, get_sparse_embedder


COLLECTION_NAME = "second_brain"
VECTOR_SIZE = 768  # nomic-embed-text output dimension
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def get_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:32768")


def _expected_hybrid_schema(client: QdrantClient) -> bool:
    """Return True if the existing collection has the dense + sparse
    named-vector config we expect. False means the collection is either
    missing one of the two vector fields or has an old/wrong shape.
    """
    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        return False

    params = getattr(info, "config", None)
    if params is None:
        return False
    p = getattr(params, "params", params)

    vectors = getattr(p, "vectors", None)
    sparse_vectors = getattr(p, "sparse_vectors", None)
    if vectors is None or sparse_vectors is None:
        return False

    # `vectors` is a dict-like mapping name -> VectorParams when named
    # vectors are used. Older collections have a single VectorParams.
    if not isinstance(vectors, dict):
        return False
    if DENSE_VECTOR_NAME not in vectors:
        return False
    if SPARSE_VECTOR_NAME not in sparse_vectors:
        return False

    dense = vectors[DENSE_VECTOR_NAME]
    if getattr(dense, "size", None) != VECTOR_SIZE:
        return False
    if getattr(dense, "distance", None) != Distance.COSINE:
        return False

    return True


def ensure_collection(force_reindex: bool = False) -> None:
    """Create the Qdrant hybrid collection (dense + sparse), or migrate
    an old/wrong schema by dropping and recreating.

    `force_reindex=True` is destructive: deletes the collection first.
    `force_reindex=False` will still drop+recreate if the existing
    collection doesn't have the expected hybrid schema (auto-migration).
    """
    client = get_client()

    existing = {c.name for c in client.get_collections().collections}
    collection_exists = COLLECTION_NAME in existing

    if collection_exists:
        if force_reindex:
            print(f"FORCE_REINDEX: dropping collection '{COLLECTION_NAME}'")
            client.delete_collection(collection_name=COLLECTION_NAME)
            collection_exists = False
        elif _expected_hybrid_schema(client):
            # Schema is already correct -> nothing to do.
            return
        else:
            # Auto-migrate: existing collection has wrong schema.
            print(
                f"Schema mismatch on '{COLLECTION_NAME}': expected hybrid "
                f"(dense + sparse). Dropping and recreating."
            )
            client.delete_collection(collection_name=COLLECTION_NAME)
            collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                ),
            },
        )
        print(f"Created hybrid collection: {COLLECTION_NAME}")


def get_existing_ids() -> set[str]:
    """Fetch all stored payload IDs to skip re-indexing."""
    client = get_client()
    existing = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["chunk_id"],
            limit=1000,
            offset=offset,
        )
        for r in results:
            if r.payload and "chunk_id" in r.payload:
                existing.add(r.payload["chunk_id"])
        if offset is None:
            break
    return existing


def index_chunks(
    chunks: list[Chunk],
    batch_size: int = 32,
    force_reindex: bool = False,
) -> None:
    """Index chunks into Qdrant w/ both dense and sparse vectors, skipping
    any chunk whose `chunk_id` is already present.
    """
    ensure_collection(force_reindex=force_reindex)
    client = get_client()
    existing_ids = set() if force_reindex else get_existing_ids()

    new_chunks = [c for c in chunks if c.id not in existing_ids]
    if not new_chunks:
        print("All chunks already indexed.")
        return

    dense = get_dense_embedder()
    sparse = get_sparse_embedder()

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        embed_texts = [f"{c.context_prefix}\n\n{c.content}" for c in batch]

        dense_vecs = dense.embed_batch(embed_texts)
        sparse_vecs = sparse.embed_batch(embed_texts)

        points = []
        for chunk, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs):
            # fastembed's SparseEmbedding has `.indices` and `.values`
            # attributes (it's a NamedTuple). Qdrant expects a
            # `SparseVector` model with the same fields.
            qdrant_sparse = SparseVector(
                indices=list(s_vec.indices),
                values=list(s_vec.values),
            )
            points.append(PointStruct(
                id=str(uuid.uuid4()),  # Qdrant point id (UUID or int)
                vector={
                    DENSE_VECTOR_NAME: d_vec,
                    SPARSE_VECTOR_NAME: qdrant_sparse,
                },
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    **chunk.metadata,
                },
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Indexed batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"Done: {len(new_chunks)} new chunks indexed.")