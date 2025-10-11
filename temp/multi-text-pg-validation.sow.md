# SOW: PostgreSQL Multi-Text Field Validation

**作成日**: 2025-10-11  
**担当**: Codex (AIアシスタント)  
**対象リポジトリ**: `MiniRAG`

---

## 1. 背景
- 2025-10-04 時点で MiniRAG の構造化インサート機能が「複数テキストフィールドを対象に検索できる」よう拡張された。
- 指示書 `Instructions1.md` に基づき、PostgreSQL スキーマを更新し、実際の PG 環境で複数テキストフィールドの挙動（登録・フィルタリング検索）を TDD 方針で検証する必要がある。

## 2. 目的
1. `postgres/sql/migrations/001_init_schema.sql` に複数テキストフィールドを持つ新テーブル定義を追加し、構造化インサートの受け皿を用意する。
2. MiniRAG の `ainsert` で `text_fields` を複数指定した場合に PostgreSQL に正しく書き込めることを確認する。
3. PostgreSQL 上のチャンク／メタデータを使ってテキストフィールド別のフィルタリング検索が行えることを自動テストで実証する。

## 3. スコープ
- **テーブル設計**: `public.customer_orders`（仮称）を追加。`doc_id` 主キーと `title`, `summary`, `body` など複数 TEXT カラム、数値・日時列、`metadata JSONB` を含める。
- **マイグレーション更新**: 上記テーブル作成と合わせて、必要な `INDEX`（例: `doc_id` 主キー、`created_at` 時系列検索用）を定義。
- **テスト実装**:
  - `minirag_app/tests/` に PostgreSQL 実機を使う統合テストを追加。
  - pytest + `pytest.mark.asyncio` で `MiniRAG` を PG バックエンド（`doc_status_storage="PGDocStatusStorage"`）に切替え、`ainsert` → フィルタリング検索までカバー。
  - `asyncpg` を用いてテーブル内容や `LIGHTRAG_DOC_CHUNKS` の `metadata->>'text_field'` を直接検証。
  - クエリ側は `MiniRAG.chunks_vdb.query(...)` で `metadata_filter={"text_field": ...}` を指定し、複数フィールド検索の動作確認を行う。
- **検証**: `docker compose up -d postgres` で起動したローカル Postgres (ホスト `localhost`, ポート `5433`) を対象にテストを実行。`uv run --link-mode=copy pytest ...` で結果確認。

## 4. 成果物
- `postgres/sql/migrations/001_init_schema.sql` 更新差分。
- 新規統合テストファイル（例: `minirag_app/tests/test_postgres_multitext.py`）。
- テスト実行ログ（pytest 成功）。

## 5. アウトオブスコープ
- MiniRAG クエリエンジン本体のアルゴリズム改修。
- 既存 PG 用ストレージ実装 (`minirag_app/minirag/kg/postgres_impl.py`) の大規模変更。
- 本番コンテナや CI/CD の設定変更。

## 6. 前提条件・依存関係
- `.env` に定義された `POSTGRES_USER=postgres_user`, `POSTGRES_PASSWORD=postgres_pass`, `POSTGRES_DB=my_database` を想定し、テスト内で利用。
- Docker と pgvector 拡張が利用可能であること。
- 既存マイグレーションに副作用を与えないよう、新テーブルは既存テーブルと衝突しない名称を採用。
- テスト実行前に Postgres コンテナが起動していること（起動責任はテスター）。

## 7. 受け入れ基準
1. マイグレーション適用後に `public.customer_orders`（仮称）が作成され、複数テキスト列を備えている。
2. 統合テストで `MiniRAG.ainsert(..., text_fields=[...])` が成功し、該当レコードが Postgres に登録される。
3. `MiniRAG.chunks_vdb.query` + `metadata_filter` を用いたテキストフィールド別検索がテストでパスする。
4. pytest 全体がグリーン。

## 8. スケジュール目安
- 設計＆マイグレーション更新: 2 時間
- テスト実装: 2 時間
- 実行＆ドキュメント整備: 1 時間

---

ご確認の上、承認の可否をご指示ください。承認後に TDD サイクル (Red → Green → Refactor) で着手します。