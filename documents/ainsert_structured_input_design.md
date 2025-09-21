# ainsert構造化入力拡張 設計メモ

## 目的
- `ainsert`が辞書レコード配列を受け取り、PostgreSQLおよびインメモリ構成で構造化フィールドを処理できるようにする。
- 従来の文字列入力、`ids`、`metadatas`の後方互換を維持する。
- 指定されたテキストフィールドをチャンク化対象として扱い、文字列リストは要素ごとにチャンク化する。

## パラメータ拡張
- `ainsert(..., schema: dict | None = None, text_fields: list[str] | None = None)` を追加。
- `schema` は以下のキーを持つ辞書を想定。
  - `table`: 完全修飾テーブル名。
  - `id_column`: 任意。省略時は `doc_id`。
  - `fields`: {フィールド名: {"type": str, "nullable": bool?}}。

## 入力データ解釈
- 受理する形式
  - 文字列 or 文字列配列（従来通り）
  - 辞書 or 辞書配列
- 辞書レコードの場合
  - `metadata` キーを取り出し既存のメタデータとマージ。
  - `schema` に従ってPostgreSQLへINSERT用の値辞書を構築。
  - テキストチャンク対象となるフィールドのみ本文を生成。リスト形式は要素ごとにチャンク化。
  - `id_column` 値を `ids` として利用。未指定の場合従来通り内部IDを生成。

## テキストチャンク生成
- 構造化レコードはテキストフィールドを抽出し、結合テキストを `DocStatus` 登録へ渡す。
- テキストフィールドがリストの場合は要素ごとにチャンク化対象のテキストとして扱う。

## PostgreSQL連携
- `schema` 指定時、PostgreSQL構成では `schema["table"]` に対するINSERT/UPSERTを実施。
- 型ごとの整形は呼び出し側指定を尊重し、asyncpgのパラメータバインドで安全に投入する。
- 衝突ポリシーは `schema.get("conflict_columns")` が存在する場合のみ `ON CONFLICT` を生成。

## インメモリ構成
- `schema` 指定時でもPostgreSQL INSERTはスキップされるため、構造化フィールドはドキュメントメタデータへ格納するのみ。

## 後方互換
- 従来の文字列入力経路は変更しない。
- `ids`/`metadatas` 引数は優先して適用し、構造化レコード側の`metadata`は補完としてマージする。

## テスト方針
- Red: `tests/test_ainsert_structured.py`
  - 構造化レコード入力で`doc_status`に正しいメタデータと本文が登録されること。
  - Postgres構成時にINSERTクエリが呼ばれること（モック）。
  - 既存文字列入力が引き続き動作すること。
- Green: 実装追加でテストをパス。
- Refactor: 正規化処理の共通化など整理。

