# MiniRAG マルチフィールド検索リファレンス

## 1. 概要

MiniRAG は構造化ドキュメントの各テキストフィールドを個別チャンクとして保存し、検索時にフィールド指定やメタデータフィルタを組み合わせた精緻な絞り込みが可能です。本書では以下を整理します。

- `MiniRAG` インスタンスの主要設定 (`enable_field_splitting`, `generate_combined_chunk`, `text_field_keys`)
- `QueryParam` のよく使うオプション (`include_provenance`, `only_need_context`, `target_fields`, `metadata_filter`)
- 同期 (`query`) / 非同期 (`aquery`) API の使い分け
- 複数フィールド検索と Provenance 付きレスポンスの挙動

## 2. MiniRAG の初期設定

フィールド分割を有効にすると、指定したテキストフィールドごとに個別チャンクが生成され、さらに統合版 `_all` チャンクも併せて作成されます。

```python
from minirag import MiniRAG

rag = MiniRAG(
    working_dir="./minirag_cache",
    enable_field_splitting=True,   # フィールド単位のチャンク生成を有効化（既定: True）
    generate_combined_chunk=True,  # 統合版（_all）チャンクも生成（既定: True）
    text_field_keys=[              # テキストフィールドとして認識するキー
        "title", "description", "summary", "content_list", "body", "text"
    ]
)
```

- `enable_field_splitting=False` にすると従来通り単一チャンク化のみ行われます。
- `generate_combined_chunk=False` にすると `_all` チャンクの生成を抑止できます。
- `text_field_keys` はリスト・文字列型の値を自動でテキストフィールド扱いするためのヒントです。

## 3. QueryParam の主要オプション

| オプション | 型 / 既定値 | 説明 |
|-------------|-------------|------|
| `mode` | `"mini"`, `"light"`, `"naive"` など | 検索モードを指定。`"mini"` は軽量回答生成、`"light"` はベクトル検索中心。 |
| `include_provenance` | `bool` (`False`) | `True` にすると回答/コンテキストと併せて `provenance` 情報（entities/chunks）を返却。 |
| `only_need_context` | `bool` (`False`) | `True` にすると回答ではなくコンテキスト文字列を返却。`include_provenance` と併用可。 |
| `target_fields` | `list[str]` (`None`) | 検索対象とする `metadata->>'text_field'` のリスト。未指定の場合 `_all` が利用されます。 |
| `metadata_filter` | `dict[str, Any]` (`None`) | チャンクの `metadata` をキー指定でフィルタ。`target_fields` と併用してさらに絞り込み。 |
| `start_time` / `end_time` | `str` (ISO 8601) | チャンクの `metadata` 内に保存されたタイムスタンプを UTC として比較し、時間範囲でフィルタ。例: `start_time="2025-10-01T00:00:00+00:00"` |
| `top_k` | `int` (`5`) | 取得するチャンク数。モードによって既定値が異なる場合があります。 |

`QueryParam` は同期 API (`MiniRAG.query`) と非同期 API (`MiniRAG.aquery`) の双方で利用できます。

## 4. 代表的な利用パターン

以下の例では `rag` は前述の設定で初期化済みとします。

### 4.1 回答と出典情報を取得する

```python
from minirag import QueryParam

param = QueryParam(mode="mini", include_provenance=True)
result, sources = rag.query("2026年度調達計画の概要は？", param=param)

# result:
# {
#   "answer": "...",
#   "provenance": {
#       "entities": [...],
#       "chunks": [...]
#   }
# }
# sources: 使用したチャンク本文のリスト
```

### 4.2 コンテキストのみ欲しい場合

```python
param = QueryParam(
    mode="mini",
    include_provenance=True,
    only_need_context=True
)
context, sources = rag.query("調達計画の詳細を教えて", param=param)

# context:
# {
#   "context": "...",
#   "provenance": {...}
# }
```

### 4.3 フィールドを限定した検索（同期）

```python
# title フィールドのみ検索
param = QueryParam(
    mode="light",
    target_fields=["title"]
)
answer, sources = rag.query("注文", param=param)

# title と description を検索
param = QueryParam(
    mode="light",
    target_fields=["title", "description"]
)
answer, sources = rag.query("注文", param=param)
```

### 4.4 フィールドを限定した検索（非同期）

```python
param = QueryParam(
    mode="light",
    target_fields=["summary"]
)
answer, sources = await rag.aquery("調達条件", param=param)
```

### 4.5 `target_fields` と `metadata_filter` の併用

```python
param = QueryParam(
    mode="light",
    target_fields=["title"],
    metadata_filter={
        "category": "order",
        "year": 2025
    },
    include_provenance=True
)
result, sources = await rag.aquery("注文", param=param)
```

### 4.6 時間フィルタ（`start_time` / `end_time`）の利用

```python
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)

param = QueryParam(
    mode="light",
    start_time=now.isoformat(),
    end_time=(now + timedelta(hours=1)).isoformat(),
    metadata_filter={"category": "plan"},
)

answer, sources = await rag.aquery("最近登録された計画について教えて", param=param)
```

- `start_time` / `end_time` は ISO 8601 文字列で指定し、UTC 基準で比較されます。
- どちらかのみ指定した場合は片側開区間となり、`metadata_filter` や `target_fields` と自由に併用できます。

## 5. 返却形式の整理

- `MiniRAG.query(...)` / `MiniRAG.aquery(...)` は常にタプル `(first, sources)` を返却します。
- `include_provenance=False` （既定）かつ `only_need_context=False` の場合、`first` は回答文字列です。
- `include_provenance=True` の場合、`first` は `dict` となり、`answer` または `context` と `provenance` を含みます。
- `only_need_context=True` の場合、`first` は `{"context": <str>, ...}` となり回答生成をスキップします。

Provenance の各要素は以下の構造を持ちます。

```json
{
  "entities": [
    {
      "entity_name": "...",
      "score": 0.87,
      "description": "..."
    }
  ],
  "chunks": [
    {
      "chunk_id": "chunk-order-2026-plan-title",
      "full_doc_id": "order-2026-plan",
      "chunk_order_index": 0,
      "tokens": 12,
      "content": "2026年度 調達計画"
    }
  ]
}
```

## 6. データ準備と検証のヒント

- `postgres/sql/migrations/002_insert_sample_data.sql` は複数テキストフィールドと `_all` チャンクを含むサンプルデータを投入します。マルチフィールド検索の動作確認に利用できます。
- `postgres/sql/migrations/001_init_schema.sql` では既存チャンクに `text_field` メタデータを付与し、`idx_chunks_text_field` インデックスを整備します。新規セットアップ時は先に実行してください。
- 自動テストは `uv run --link-mode=copy pytest minirag_app/tests/test_postgres_multi_field_search.py` で実行できます。Provenance 返却や `metadata_filter` との複合条件をカバーしています。

## 7. トラブルシューティング

| 症状 | 確認ポイント |
|------|--------------|
| `target_fields` を指定しても結果が空になる | マイグレーションで `metadata->>'text_field'` が `_all` も含めて設定されているか確認。 |
| `metadata_filter` が効かない | 値が文字列化されているか、キー名のスペルが一致しているか確認。必要なら `jsonb` の中身を `SELECT metadata FROM LIGHTRAG_DOC_CHUNKS ...` で検証。 |
| Provenance が返ってこない | `include_provenance=True` を設定しているか、`mode` が `mini` 以外の場合はドキュメント処理が完了しているかを確認。 |

---

最新の仕様に沿ってサンプルコードを整理しました。既存のクイックスタートや Notebook と併用し、マルチフィールド検索の検証にご利用ください。

