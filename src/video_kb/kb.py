from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import DependencyError, load_dotenv_if_available, load_json, save_json


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class Embedder:
    provider: str
    model_name: str
    dimension: int
    model: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "hash":
            return [hash_embedding(text, self.dimension) for text in texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]


def _tokens(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    chinese = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [f"{chinese[i]}{chinese[i + 1]}" for i in range(len(chinese) - 1)]
    return tokens + bigrams


def hash_embedding(text: str, dimension: int = 384) -> list[float]:
    vector = [0.0] * dimension
    tokens = _tokens(text) or [text[:64] or "empty"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dimension
        sign = 1.0 if value & (1 << 63) else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def build_embedder(
    provider: str = "sentence-transformers",
    model_name: str | None = None,
    fallback_hash: bool = True,
    hash_dimension: int = 384,
) -> Embedder:
    load_dotenv_if_available()
    model_name = model_name or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-zh-v1.5"
    if provider == "hash":
        return Embedder(provider="hash", model_name="hashing-vectorizer", dimension=hash_dimension)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        dimension = int(model.get_sentence_embedding_dimension())
        return Embedder(
            provider="sentence-transformers",
            model_name=model_name,
            dimension=dimension,
            model=model,
        )
    except Exception as exc:
        if not fallback_hash:
            raise DependencyError(
                "sentence-transformers or the embedding model could not be loaded."
            ) from exc
        return Embedder(provider="hash", model_name="hashing-vectorizer", dimension=hash_dimension)


def _point_text(point: dict[str, Any]) -> str:
    return "\n".join(
        str(point.get(key, "") or "")
        for key in ["title", "summary", "evidence_text", "speaker"]
    ).strip()


def _docs_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}_docs.json")


def _meta_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}_meta.json")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_kb(
    summary_path: str | Path,
    db_path: str | Path,
    collection_name: str = "video_knowledge",
    backend: str = "qdrant",
    embedding_provider: str = "sentence-transformers",
    embedding_model: str | None = None,
    fallback_hash: bool = True,
    source_file: str | None = None,
    meta_output_path: str | Path | None = None,
) -> dict[str, Any]:
    if backend not in {"qdrant", "json", "milvus"}:
        raise ValueError("backend must be one of: qdrant, json, milvus")

    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    summary = load_json(summary_path)
    points = summary.get("knowledge_points", [])
    embedder = build_embedder(
        provider=embedding_provider,
        model_name=embedding_model,
        fallback_hash=fallback_hash,
    )

    texts = [_point_text(point) for point in points]
    vectors = embedder.encode(texts) if texts else []
    rows = []
    docs = []
    for idx, point in enumerate(points):
        tags = point.get("tags", [])
        tags_text = ",".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags or "")
        payload = {
            "id": idx,
            "title": str(point.get("title", "") or ""),
            "summary": str(point.get("summary", "") or ""),
            "evidence_text": str(point.get("evidence_text", "") or ""),
            "start_ms": _safe_int(point.get("start_ms")),
            "end_ms": _safe_int(point.get("end_ms")),
            "speaker": str(point.get("speaker", "") or "UNKNOWN"),
            "tags": tags_text,
            "source_file": source_file or str(summary_path),
        }
        row = {"id": idx, "vector": vectors[idx], **payload}
        rows.append(row)
        docs.append(dict(row))

    actual_backend = "json"
    backend_error = None
    if backend == "qdrant":
        try:
            _write_qdrant(db, collection_name, embedder.dimension, rows)
            actual_backend = "qdrant"
        except Exception as exc:
            if not fallback_hash:
                raise
            backend_error = str(exc)
    elif backend == "milvus":
        try:
            _write_milvus(db, collection_name, embedder.dimension, rows)
            actual_backend = "milvus"
        except Exception as exc:
            if not fallback_hash:
                raise
            backend_error = str(exc)

    docs_path = _docs_path(db)
    save_json({"records": docs}, docs_path)
    meta = {
        "summary_path": str(summary_path),
        "db_path": str(db),
        "docs_path": str(docs_path),
        "collection_name": collection_name,
        "requested_backend": backend,
        "backend": actual_backend,
        "backend_error": backend_error,
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "record_count": len(rows),
    }
    save_json(meta, meta_output_path or _meta_path(db))
    return meta


def _write_qdrant(
    db_path: Path,
    collection_name: str,
    dimension: int,
    rows: list[dict[str, Any]],
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client import models

    client = QdrantClient(path=str(db_path))
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        if rows:
            points = []
            for row in rows:
                payload = {key: value for key, value in row.items() if key not in {"id", "vector"}}
                points.append(
                    models.PointStruct(
                        id=int(row["id"]),
                        vector=row["vector"],
                        payload=payload,
                    )
                )
            client.upsert(collection_name=collection_name, points=points, wait=True)
    finally:
        if hasattr(client, "close"):
            client.close()


def _write_milvus(
    db_path: Path,
    collection_name: str,
    dimension: int,
    rows: list[dict[str, Any]],
) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(str(db_path))
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    try:
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            metric_type="COSINE",
        )
    except TypeError:
        client.create_collection(collection_name=collection_name, dimension=dimension)
    if rows:
        client.insert(collection_name=collection_name, data=rows)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity", {})
        return {
            "score": hit.get("distance", hit.get("score")),
            **entity,
            "id": hit.get("id", entity.get("id")),
        }
    entity = getattr(hit, "entity", {}) or {}
    return {
        "score": getattr(hit, "distance", None),
        **dict(entity),
        "id": getattr(hit, "id", entity.get("id") if isinstance(entity, dict) else None),
    }


def _qdrant_point_to_dict(point: Any) -> dict[str, Any]:
    payload = dict(getattr(point, "payload", None) or {})
    score = getattr(point, "score", None)
    point_id = getattr(point, "id", payload.get("id"))
    return {"id": point_id, **payload, "score": score}


def _query_qdrant(
    db_path: Path,
    collection_name: str,
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(db_path))
    try:
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(result, "points", result)
        return [_qdrant_point_to_dict(point) for point in points]
    finally:
        if hasattr(client, "close"):
            client.close()


def _query_milvus(
    db_path: Path,
    collection_name: str,
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    from pymilvus import MilvusClient

    client = MilvusClient(str(db_path))
    hits = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=limit,
        output_fields=[
            "title",
            "summary",
            "evidence_text",
            "start_ms",
            "end_ms",
            "speaker",
            "tags",
            "source_file",
        ],
    )
    return [_hit_to_dict(hit) for hit in hits[0]]


def _query_json_fallback(meta: dict[str, Any], query_vector: list[float], limit: int) -> list[dict[str, Any]]:
    docs_path = Path(meta["docs_path"])
    docs = load_json(docs_path).get("records", [])
    scored = []
    for doc in docs:
        vector = doc.get("vector", [])
        scored.append((_cosine(query_vector, vector), doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, doc in scored[:limit]:
        clean = {k: v for k, v in doc.items() if k != "vector"}
        clean["score"] = score
        results.append(clean)
    return results


def query_kb(
    db_path: str | Path,
    query: str,
    limit: int = 5,
    meta_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    db = Path(db_path)
    meta = load_json(meta_path or _meta_path(db))
    embedder = build_embedder(
        provider=meta.get("embedding_provider", "sentence-transformers"),
        model_name=meta.get("embedding_model"),
        fallback_hash=True,
        hash_dimension=int(meta.get("embedding_dimension", 384)),
    )
    query_vector = embedder.encode([query])[0]
    backend = meta.get("backend")
    if backend == "qdrant":
        try:
            return _query_qdrant(db, meta["collection_name"], query_vector, limit)
        except Exception:
            pass
    if backend == "milvus":
        try:
            return _query_milvus(db, meta["collection_name"], query_vector, limit)
        except Exception:
            pass
    return _query_json_fallback(meta, query_vector, limit)
