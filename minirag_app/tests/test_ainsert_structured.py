import asyncio
from typing import Any

import pytest

from minirag_app.minirag.minirag import MiniRAG


@pytest.fixture()
def mini_rag(tmp_path):
    rag = MiniRAG(working_dir=str(tmp_path))
    rag.chunking_func = lambda content, overlap, size, model: [
        {
            "content": content,
            "chunk_order_index": 0,
            "tokens": len(content),
        }
    ]
    return rag


@pytest.fixture()
def structured_records() -> list[dict[str, Any]]:
    return [
        {
            "doc_id": "order-001",
            "title": "注文1",
            "price": 123.45,
            "cnt": 7,
            "description": ["長文A", "長文B"],
            "created_at": "2025-09-01T12:00:00Z",
            "metadata": {"category": "order", "country": "Japan", "year": 2025},
        }
    ]


@pytest.fixture()
def structured_schema() -> dict[str, Any]:
    return {
        "table": "public.customer_orders",
        "id_column": "doc_id",
        "fields": {
            "doc_id": {"type": "text", "nullable": False},
            "title": {"type": "text"},
            "price": {"type": "float"},
            "cnt": {"type": "integer"},
            "description": {"type": "text"},
            "created_at": {"type": "timestamp"},
        },
    }


@pytest.mark.asyncio
async def test_ainsert_structured_records_merge_metadata(mini_rag, structured_records, structured_schema):
    await mini_rag.ainsert(
        structured_records,
        schema=structured_schema,
        text_fields=["title", "description"],
    )

    stored_docs = mini_rag.doc_status._data
    doc_id = structured_records[0]["doc_id"]
    assert doc_id in stored_docs

    metadata = stored_docs[doc_id]["metadata"]
    assert metadata["category"] == "order"
    assert metadata["country"] == "Japan"
    assert metadata["price"] == 123.45
    assert metadata["cnt"] == 7
    assert metadata["year"] == 2025

    content = stored_docs[doc_id]["content"]
    assert "注文1" in content
    assert "長文A" in content
    assert "長文B" in content

    # 確認: チャンク側にも構造化メタデータが引き継がれる
    chunk_values = list(mini_rag.text_chunks._data.values())
    assert chunk_values, "チャンクが生成されていません"
    for chunk in chunk_values:
        assert chunk["metadata"]["category"] == "order"
        assert chunk["metadata"]["price"] == 123.45


@pytest.mark.asyncio
async def test_ainsert_invokes_postgres_writer(monkeypatch, mini_rag, structured_records, structured_schema):
    async def fake_pg_writer(records, schema):
        fake_pg_writer.called = True
        fake_pg_writer.records = records
        fake_pg_writer.schema = schema

    fake_pg_writer.called = False
    monkeypatch.setattr(mini_rag, "_write_structured_records_to_pg", fake_pg_writer)
    monkeypatch.setattr(mini_rag, "_is_postgres_backend", lambda: True)

    await mini_rag.ainsert(
        structured_records,
        schema=structured_schema,
        text_fields=["title", "description"],
    )

    assert fake_pg_writer.called, "PostgreSQL書き込み処理が呼び出されませんでした"
    assert fake_pg_writer.records[0]["title"] == "注文1"
    assert fake_pg_writer.schema["table"] == "public.customer_orders"

