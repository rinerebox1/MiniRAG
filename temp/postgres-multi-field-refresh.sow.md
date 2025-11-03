# SOW: PostgreSQL マルチフィールド対応リフレッシュ

**作成日**: 2025-11-03  
**プロジェクト**: MiniRAG  
**作業対象**: PostgreSQL向けマルチフィールド検索周りの再整備

---

## 1. 背景と目的

- `001_init_schema.sql` で初期スキーマを整備したが、後続マイグレーション `002_add_text_field_to_existing_chunks.sql` が別管理となっており、新規環境では `_all` チャンク生成や `text_field` インデックスが未適用のままになる。
- `002_insert_sample_data.sql` は古い構造（単一テキストフィールド前提）を前提にしており、構造化データとフィールド別チャンクの挙動を確認しづらい。
- ユーザー向け利用例が複数箇所に散在し、`QueryParam` の `include_provenance` / `only_need_context` / `target_fields` / `metadata_filter` の正しい使い方が把握しづらい。
- Notebook `MiniRAG_on_postgres.ipynb` でも、複数フィールド挿入とフィールド絞り込み検索の最新シナリオを再現できていない。

目的は以下の通り：

1. 新規セットアップ時点で `text_field` メタデータとインデックスが必ず揃うようにマイグレーションを一元化する。
2. サンプルデータを最新仕様（構造化ドキュメント・複数テキストフィールド）に合わせ、再現性の高い挙動確認を可能にする。
3. ユーザーが `QueryParam` のオプションと `MiniRAG` 初期化パラメータを理解できる参照ドキュメントを用意する。
4. Notebook 上で複数フィールド挿入・検索・Provenance 返却を通しで検証できるようにする。

---

## 2. スコープ

### 2.1 対象作業

1. **マイグレーション統合**
   - `postgres/sql/migrations/001_init_schema.sql` に `text_field` 補完ロジックと `idx_chunks_text_field` 作成処理を組み込み、新規環境でも即時利用可能にする。

2. **サンプルデータ刷新**
   - `postgres/sql/migrations/002_insert_sample_data.sql` を、複数テキストフィールド（`title`, `description`, `summary`, `body`, `_all`）および `metadata` を持つサンプルに更新。
   - `customer_orders` テーブルと `LIGHTRAG_DOC_CHUNKS` 両方に挿入するサンプルを準備し、pgvector・AGE 初期化手順を現状仕様に整合させる。

3. **ドキュメント整理**
   - `documents/MiniRAG_multi_field_query.md`（新規）を作成し、以下を網羅：
     - `MiniRAG` 初期化オプション (`enable_field_splitting`, `generate_combined_chunk`, `text_field_keys`)
     - `QueryParam` の主要オプションと返却値 (`include_provenance`, `only_need_context`, `target_fields`, `metadata_filter`, `mode`)
     - 同期/非同期 `query`/`aquery` の使用例、Provenance 付きレスポンス例
     - `target_fields` と `metadata_filter` の組み合わせ挙動、デフォルト `_all` の説明

4. **Notebook 更新**
   - `minirag_app/docs/MiniRAG_on_postgres.ipynb` にて、
     1. 構造化データ挿入 (`ainsert`) で複数フィールドを登録するセル
     2. `target_fields` 指定検索と `_all` デフォルト検索の比較セル
     3. `metadata_filter` と併用する検索セル
     4. `include_provenance` + `only_need_context` の返却例
     を追加または差し替えし、差分が追いかけやすいよう Markdown 解説も整理する。

5. **検証**
   - 主要テスト: `uv run --link-mode=copy pytest minirag_app/tests/test_postgres_multi_field_search.py`
   - Notebook の手動実行手順と確認項目をドキュメント化（セル末尾コメント等）。

### 2.2 非スコープ

- 既存データの再マイグレーションは利用者判断とし、スクリプトの提供のみ。
- API 仕様や OpenAPI への変更。
- 新たなテストケース追加以外の既存ビジネスロジック改修。

---

## 3. 成果物

| カテゴリ | ファイル / 成果物 | 内容 |
|---|---|---|
| マイグレーション | `postgres/sql/migrations/001_init_schema.sql` | `LIGHTRAG_DOC_CHUNKS` への `text_field` 初期化 + インデックス統合 |
| マイグレーション | `postgres/sql/migrations/002_insert_sample_data.sql` | 構造化サンプルデータおよび複数フィールドチャンク挿入 |
| ドキュメント | `documents/MiniRAG_multi_field_query.md` | `QueryParam` / `MiniRAG` 利用ガイド |
| Notebook | `minirag_app/docs/MiniRAG_on_postgres.ipynb` | 複数フィールド検索シナリオの検証手順 |
| 検証記録 | テスト結果ログ / Notebook 実行結果 | pytest 成功ログ、Notebookの確認メモ |

---

## 4. タスク分解と見積

| ID | タスク | 詳細 | 見積時間 |
|---|---|---|---|
| T1 | マイグレーション統合 | `001_init_schema.sql` に `text_field` 補完とインデックス作成を統合 | 1.5 h |
| T2 | サンプルデータ刷新 | `002_insert_sample_data.sql` を構造化サンプルに刷新 | 2.0 h |
| T3 | ドキュメント追加 | `MiniRAG_multi_field_query.md` で利用例整理 | 1.5 h |
| T4 | Notebook 更新 | ノートブックに複数テキストフィールド対応の検証セルを追加 | 3.0 h |
| T5 | 検証 | pytest 実行、Notebook 動作確認、差分レビュー | 1.0 h |

---

## 5. 依存関係

- T2 は T1 と並行可能だが、`_all` 補完仕様を踏まえてサンプル構造を決定するため、T1の方針確定後に着手。
- T3/T4 は最新仕様（T1/T2）確定後に作成。
- T5 は全更新完了後に実施。

---

## 6. リスクと緩和策

| リスク | 影響 | 緩和策 |
|---|---|---|
| 既存データの `metadata` 形式が想定外 | 既存環境で `jsonb_set` が失敗 | UPDATE 前に `COALESCE(metadata,'{}')` を用い、例外時にはログ出力とマニュアル手順を記載 |
| サンプルデータ挿入による重複キー | マイグレーション失敗 | `ON CONFLICT DO NOTHING` / `DELETE` 事前実行で回避 |
| Notebook セル出力の非決定性 | ドキュメントと動作が乖離 | 疑似埋め込み・モック LLM を使用し、決定的な結果を返すよう設定 |
| pytest 実行環境の差異 | CI 失敗 | `uv` と `--link-mode=copy` を手順に明記 |

---

## 7. 成功基準

- 新規構築時に `LIGHTRAG_DOC_CHUNKS.metadata->>'text_field'` が `_all` を含み、インデックスが存在する。
- サンプルデータ挿入後に `title`/`summary`/`body` 等のチャンクが作成され、`target_fields` 指定検索が機能する。
- `documents/MiniRAG_multi_field_query.md` を参照すれば、`QueryParam` と `MiniRAG` 初期化オプションの使用法が明確になる。
- Notebook で複数フィールド検索・Provenance 返却が再現でき、手順がセル内で説明される。
- `pytest minirag_app/tests/test_postgres_multi_field_search.py` が成功する。

---

## 8. スケジュール（目安）

| 日付 | 作業 | 備考 |
|---|---|---|
| Day 0 | SOW承認 | 本ドキュメント |
| Day 1 | T1, T2 完了 | マイグレーション統合＋サンプルデータ刷新 |
| Day 2 | T3, T4 完了 | ドキュメント・Notebook 更新 |
| Day 3 | T5 完了 | pytest と Notebook 検証、レビュー反映 |

---

## 9. 承認ポイント

1. スコープ（T1〜T5）が要求を満たしているか。
2. 成果物の配置パス（migrations/documents/notebook）が妥当か。
3. 検証方法（pytest + Notebook）で十分か。
4. スケジュールと見積時間が現実的か。

---

承認後、T1（マイグレーション統合）から着手します。

