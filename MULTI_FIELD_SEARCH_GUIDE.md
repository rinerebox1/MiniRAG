# Multi-Field Search 機能ガイド

2025/10/04

## 概要

MiniRAGに複数テキストフィールドに対する柔軟な検索機能が追加されました。
構造化データ（dict）の各フィールドを個別にチャンク化し、フィールド単位での検索が可能になります。

## 主な機能

### 1. フィールド別チャンク生成

入力データの各テキストフィールドごとに個別のチャンクを生成します。

```python
data = {
    "doc_id": "order-001",
    "title": "注文1",
    "description": ["長文A", "長文B"],
    "price": 123.45,
    "metadata": {"category": "order"}
}

await rag.ainsert([data])
```

生成されるチャンク:
- `title` フィールドのチャンク → `metadata: {"text_field": "title"}`
- `description` フィールドのチャンク → `metadata: {"text_field": "description"}`
- 統合版チャンク（デフォルト） → `metadata: {"text_field": "_all"}`

### 2. フィールド指定検索

`target_fields` パラメータで検索対象フィールドを指定できます。

```python
from minirag import QueryParam

# titleフィールドのみ検索
param = QueryParam(
    mode="light",
    target_fields=["title"]
)
answer, source = await rag.aquery("注文", param=param)

# title と description を検索
param = QueryParam(
    mode="light",
    target_fields=["title", "description"]
)
answer, source = await rag.aquery("注文", param=param)

# デフォルト（統合検索、_all）
# title と description が結合されたテキスト content があり、content を検索する
param = QueryParam(mode="light")
answer, source = await rag.aquery("注文", param=param)
```

### 3. メタデータフィルタとの併用

`target_fields` と `metadata_filter` を組み合わせて使用できます。

```python
# title テキストを対象に検索しつつ、さらに、その中から metadata_filter で絞られる
param = QueryParam(
    mode="light",
    target_fields=["title"],
    metadata_filter={"category": "order", "year": 2025}
)
answer, source = await rag.aquery("注文", param=param)
```

## 設定オプション

MiniRAGインスタンス作成時に以下の設定が可能です：

```python
rag = MiniRAG(
    working_dir="./minirag_cache",
    enable_field_splitting=True,  # フィールド分割の有効化（デフォルト: True）
    generate_combined_chunk=True,  # 統合版チャンクの生成（デフォルト: True）
    text_field_keys=["title", "description", "summary", "content_list", "body", "text"]
)
```

### `text_field_keys`

自動的にテキストフィールドとして認識されるキー名のリスト。
これらのキーまたは `str`/`list` 型の値は自動的にテキストフィールドとして扱われます。

## 既存データのマイグレーション

既存のチャンクデータに `text_field="_all"` を追加するマイグレーションスクリプトを用意しています。

```bash
psql -h your-db-host -U your-user -d your-database -f 002_add_text_field_to_existing_chunks.sql
```

または、Dockerコンテナ内から：

```bash
psql -U postgres -d minirag -f /path/to/002_add_text_field_to_existing_chunks.sql
```

## 技術的な詳細

### データフロー

1. **チャンク生成時**（`apipeline_process_enqueue_documents`）
   - 構造化データの場合、各フィールドごとに個別のチャンクを生成
   - 各チャンクのメタデータに `text_field` を付与
   - 統合版チャンク（`_all`）も生成

2. **検索時**（`aquery`）
   - `target_fields` が指定されている場合、内部で `metadata_filter` に変換
   - PostgreSQLのメタデータフィルタでフィールドを絞り込み
   - IN句を使った複数フィールド検索をサポート

### PostgreSQLクエリ例

```sql
-- 単一フィールド検索
SELECT * FROM LIGHTRAG_DOC_CHUNKS
WHERE metadata->>'text_field' = 'title';

-- 複数フィールド検索
SELECT * FROM LIGHTRAG_DOC_CHUNKS
WHERE metadata->>'text_field' IN ('title', 'description');

-- デフォルト検索（統合版）
SELECT * FROM LIGHTRAG_DOC_CHUNKS
WHERE metadata->>'text_field' = '_all';
```

## 後方互換性

- `target_fields` を指定しない場合、デフォルトで `_all`（統合版）が検索されます
- 既存のコードは変更なしで動作します
- マイグレーションスクリプトにより、既存データにも `text_field="_all"` が追加されます

## パフォーマンス考慮事項

### ストレージ増加

各フィールドごとにチャンクを生成するため、ストレージ使用量が増加します。
- `title` + `description` の場合：約 2〜3倍
- `generate_combined_chunk=False` にすることで統合版を無効化可能

### インデックス

`metadata->>'text_field'` にインデックスが自動作成され、検索パフォーマンスが最適化されます。

## トラブルシューティング

### Q: 既存データで `target_fields` を指定しても結果が返ってこない

A: マイグレーションスクリプト（`002_add_text_field_to_existing_chunks.sql`）を実行してください。

### Q: ストレージ使用量が増加しすぎる

A: 以下の設定で調整可能です：
```python
rag = MiniRAG(
    generate_combined_chunk=False,  # 統合版を無効化
    enable_field_splitting=False,   # フィールド分割を完全に無効化
)
```

## 更新履歴

- 2025-10-03: Multi-Field Search 機能追加
  - フィールド別チャンク生成
  - `target_fields` パラメータ追加
  - PostgreSQL IN句サポート
  - マイグレーションスクリプト提供
