import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from minirag_app.minirag.base import QueryParam
from minirag_app.minirag.kg.postgres_impl import PGDocStatusStorage, PostgreSQLDB
import minirag_app.minirag.kg.postgres_impl as postgres_module
from minirag_app.minirag.minirag import MiniRAG
from minirag_app.minirag.utils import EmbeddingFunc

import asyncpg
import minirag_app.minirag.minirag as minirag_module
import minirag_app.minirag as minirag_package
import minirag_app.minirag.utils as minirag_utils_module
import sys

sys.modules.setdefault("minirag", minirag_package)
sys.modules.setdefault("minirag.utils", minirag_utils_module)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres_pass")
POSTGRES_DB = os.getenv("POSTGRES_DB", "my_database")


async def _wait_for_database(timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = await asyncpg.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                database=POSTGRES_DB,
            )
            await conn.close()
            return
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError("PostgreSQL did not become ready within timeout window.")


@pytest.fixture(scope="session")
def postgres_container():
    subprocess.run(
        ["docker", "compose", "up", "-d", "postgres"],
        check=True,
        cwd=PROJECT_ROOT,
    )
    asyncio.run(_wait_for_database())
    yield


@pytest_asyncio.fixture
async def pg_conn(postgres_container):
    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
    )
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
def stub_extract_entities(monkeypatch):
    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(minirag_module, "extract_entities", _noop)


@pytest.fixture(autouse=True)
def ensure_metadata_dict(monkeypatch):
    original_get = PGDocStatusStorage.get_docs_by_status

    async def _wrapped(self, status):
        result = await original_get(self, status)
        for doc in result.values():
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, str):
                doc.metadata = json.loads(metadata)
        return result

    monkeypatch.setattr(PGDocStatusStorage, "get_docs_by_status", _wrapped)


@pytest.fixture(autouse=True)
def serialize_datetimes_in_jsonb(monkeypatch):
    original_jsonb = postgres_module._jsonb

    def _jsonb_with_datetime(obj):
        if obj is None:
            return "{}"
        if isinstance(obj, str):
            return obj

        def _default(value):
            if isinstance(value, datetime):
                return value.isoformat()
            return value

        return json.dumps(obj, default=_default)

    monkeypatch.setattr(postgres_module, "_jsonb", _jsonb_with_datetime)


def _build_embedding_func() -> EmbeddingFunc:
    async def _embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            norm = max(len(text), 1)
            vectors.append(
                np.array([float(norm % 7 + 1), float(norm % 5 + 1), 0.5], dtype=np.float32)
            )
        return np.vstack(vectors)

    return EmbeddingFunc(
        embedding_dim=3,
        max_token_size=2048,
        func=_embed,
    )


@pytest_asyncio.fixture
async def mini_rag(tmp_path):
    os.environ.setdefault("COSINE_THRESHOLD", "-1.0")

    workspace = f"test_workspace_{uuid4().hex}"
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    async def _llm_stub(*args, **kwargs):
        return ""

    rag = MiniRAG(
        working_dir=str(work_dir),
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="NetworkXStorage",
        embedding_func=_build_embedding_func(),
        llm_model_func=_llm_stub,
        llm_model_max_async=1,
    )

    rag.chunking_func = lambda content, overlap, size, model: [
        {"content": content, "chunk_order_index": 0, "tokens": len(content)}
    ]
    rag.max_parallel_insert = 1

    pg_db = PostgreSQLDB(
        {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "database": POSTGRES_DB,
            "workspace": workspace,
        }
    )
    await pg_db.initdb()
    await pg_db.check_tables()
    rag.set_storage_client(pg_db)

    yield rag, pg_db, workspace

    for table in [
        "LIGHTRAG_DOC_STATUS",
        "LIGHTRAG_DOC_CHUNKS",
        "LIGHTRAG_DOC_FULL",
        "LIGHTRAG_VDB_ENTITY",
        "LIGHTRAG_VDB_RELATION",
    ]:
        await pg_db.execute(
            f"DELETE FROM {table} WHERE workspace=$1", {"workspace": workspace}
        )
    await pg_db.execute(
        "DELETE FROM public.customer_orders WHERE workspace=$1",
        {"workspace": workspace},
    )

    await pg_db.pool.close()


@pytest.mark.asyncio
async def test_structured_insert_persists_multi_text_fields(mini_rag, pg_conn):
    rag, _, workspace = mini_rag
    doc_id = "order-2025-001"
    body_sections = ["フェーズ1: 調達計画", "フェーズ2: サプライヤ連絡"]

    created_at = datetime(2025, 10, 1, 8, 30, tzinfo=timezone.utc)
    record = {
        "workspace": workspace,
        "doc_id": doc_id,
        "title": "2026年度 調達計画",
        "summary": "調達戦略の概要をまとめたサマリーです。",
        "body": body_sections,
        "status": "draft",
        "region": "APAC",
        "priority": 2,
        "created_at": created_at,
        "metadata": {
            "category": "plan",
            "region": "APAC",
            "owner": "Procurement",
            "created_at": created_at.isoformat(),
        },
    }
    schema = {
        "table": "public.customer_orders",
        "id_column": "doc_id",
        "fields": {
            "workspace": {"type": "text", "nullable": False},
            "doc_id": {"type": "text", "nullable": False},
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "body": {"type": "text"},
            "status": {"type": "text"},
            "region": {"type": "text"},
            "priority": {"type": "integer"},
            "created_at": {"type": "timestamp"},
        },
        "conflict_columns": ["workspace", "doc_id"],
    }

    await rag.ainsert([record], schema=schema, text_fields=["title", "summary", "body"])

    row = await pg_conn.fetchrow(
        """
        SELECT title, summary, body, status, region, priority
        FROM public.customer_orders
        WHERE workspace=$1 AND doc_id=$2
        """,
        workspace,
        doc_id,
    )

    assert row is not None
    assert row["title"] == record["title"]
    assert row["summary"] == record["summary"]
    assert row["body"] == "\n".join(body_sections)
    assert row["status"] == "draft"
    assert row["region"] == "APAC"
    assert row["priority"] == 2


@pytest.mark.asyncio
async def test_field_specific_query_with_metadata_filter(mini_rag, pg_conn):
    rag, _, workspace = mini_rag
    doc_id = "order-2025-002"

    created_at = datetime(2025, 10, 2, 9, 15, tzinfo=timezone.utc)
    record = {
        "workspace": workspace,
        "doc_id": doc_id,
        "title": "北米向けサプライ契約",
        "summary": "北米市場での調達条件を整理したサマリー。",
        "body": [
            "詳細1: 北米主要ベンダーの評価結果。",
            "詳細2: リスクと緩和策の一覧。",
        ],
        "status": "approved",
        "region": "APAC",
        "priority": 1,
        "created_at": created_at,
        "metadata": {
            "category": "supply",
            "region": "APAC",
            "status": "approved",
            "created_at": created_at.isoformat(),
        },
    }
    schema = {
        "table": "public.customer_orders",
        "id_column": "doc_id",
        "fields": {
            "workspace": {"type": "text", "nullable": False},
            "doc_id": {"type": "text", "nullable": False},
            "title": {"type": "text"},
            "summary": {"type": "text"},
            "body": {"type": "text"},
            "status": {"type": "text"},
            "region": {"type": "text"},
            "priority": {"type": "integer"},
            "created_at": {"type": "timestamp"},
        },
        "conflict_columns": ["workspace", "doc_id"],
    }

    await rag.ainsert([record], schema=schema, text_fields=["title", "summary", "body"])

    summary_chunk_ids = await pg_conn.fetch(
        """
        SELECT id
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE workspace=$1
          AND full_doc_id=$2
          AND metadata->>'text_field' = 'summary'
          AND metadata->>'region' = 'APAC'
        """,
        workspace,
        doc_id,
    )
    assert summary_chunk_ids, "Summary chunk with region metadata not found"

    query_param = QueryParam(
        mode="naive",
        target_fields=["summary"],
        metadata_filter={"region": "APAC"},
        only_need_context=True,
        top_k=5,
    )
    context, sources = await rag.aquery("調達条件", param=query_param)

    assert "北米市場での調達条件" in context
    assert all("北米向けサプライ契約" not in src for src in sources)
    assert sources, "Expected context sources for summary field query"

    no_match_param = QueryParam(
        mode="naive",
        target_fields=["summary"],
        metadata_filter={"region": "EMEA"},
        only_need_context=True,
        top_k=5,
    )
    no_match_context, no_match_sources = await rag.aquery("調達条件", param=no_match_param)

    assert no_match_sources == []
    assert "Sorry" in no_match_context
