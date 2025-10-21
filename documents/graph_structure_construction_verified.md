---
title: グラフ構築手順の実装検証と補足
---

以下は、提示ドキュメント（原文）に対してコードベースを用いた事実検証を行い、注意点と実装差異を追記したものです。

## 原文（参照）

コードを確認したところ、グラフ構造は以下の手順で作成されます。

## グラフ構造の構築手順

1.  **テキストのチャンク化**:
    * 入力されたテキストは、まず指定されたトークンサイズ (`chunk_token_size`) とオーバーラップ (`chunk_overlap_token_size`) に基づいて、小さなテキストチャンクに分割されます。 `chunking_by_token_size` 関数 (または `MiniRAG` クラスで指定されたカスタム関数 `chunking_func`) がこの処理を担当します。

2.  **エンティティとリレーションシップの抽出**:
    * 分割された各チャンクに対して、LLM (Large Language Model) を使用してエンティティ (ノード) とリレーションシップ (エッジ) を抽出します。 `extract_entities` 関数内で、LLM に対してエンティティ抽出用のプロンプト (`PROMPTS["entity_extraction"]`) が与えられます。
    * LLM の応答は解析され、`_handle_single_entity_extraction` 関数がエンティティ情報 (名前、タイプ、説明、ソースID=チャンクキー) を抽出し、`_handle_single_relationship_extraction` 関数がリレーションシップ情報 (ソースエンティティ、ターゲットエンティティ、説明、キーワード、ソースID=チャンクキー、重み) を抽出します。

3.  **ノードのマージと挿入/更新 (Upsert)**:
    * 抽出されたエンティティは、エンティティ名をキーとしてグループ化されます。
    * 同じエンティティ名を持つノードが複数抽出された場合、または既存のグラフに同じ名前のノードが存在する場合、`_merge_nodes_then_upsert` 関数によって情報がマージされます。
        * **エンティティタイプ**: 最も頻繁に出現したタイプが選択されます。
        * **説明 (description)**: 重複を除いて結合されます。
        * **ソースID (source_id)**: 重複を除いて結合されます。
    * マージされたノードデータは、グラフストレージ (`knowledge_graph_inst`) の `upsert_node` メソッドを使用してグラフに挿入または更新されます。
        * NetworkX の場合: `_graph.add_node(node_id, **node_data)` が呼び出されます。
        * Neo4j の場合: `MERGE (n:`{label}`) SET n += $properties` という Cypher クエリが実行されます。

4.  **エッジのマージと挿入/更新 (Upsert)**:
    * 抽出されたリレーションシップは、(ソースエンティティ, ターゲットエンティティ) のペアをキーとしてグループ化されます。
    * 同じエンティティペア間のリレーションシップが複数抽出された場合、または既存のグラフに同じエッジが存在する場合、`_merge_edges_then_upsert` 関数によって情報がマージされます。
        * **重み (weight)**: 合計されます。
        * **説明 (description)**: 重複を除いて結合されます。
        * **キーワード (keywords)**: 重複を除いて結合されます。
        * **ソースID (source_id)**: 重複を除いて結合されます。
    * マージされたエッジデータは、グラフストレージ (`knowledge_graph_inst`) の `upsert_edge` メソッドを使用してグラフに挿入または更新されます。
        * NetworkX の場合: `_graph.add_edge(source_node_id, target_node_id, **edge_data)` が呼び出されます。
        * Neo4j の場合: `MATCH (source:`{source_node_label}`) ... MATCH (target:`{target_node_label}`) MERGE (source)-[r:DIRECTED]->(target) SET r += $properties` という Cypher クエリが実行されます。
        * **ノードの存在確認**: エッジを追加する際に、対応するソースノードまたはターゲットノードが存在しない場合は、自動的に作成されます (`entity_type` は `"UNKNOWN"` として設定されます)。

5.  **ベクトルデータベースへの保存**:
    * 抽出・マージされたエンティティとリレーションシップの情報は、それぞれのベクトルデータベース (VDB) にも保存され、検索可能になります (`entity_vdb`, `entity_name_vdb`, `relationships_vdb`)。

---

## 補足・実装差異（コード検証より）

- **チャンク化の詳細**
  - 実装は `tiktoken` を用いたトークン単位の分割で、ステップ幅は `(max_token_size - overlap_token_size)`。返却には `tokens`, `content`, `chunk_order_index` が含まれます。
  - 参照: `minirag/operate.py` の `chunking_by_token_size`。

- **抽出プロンプトとパーサ整合性**
  - プロンプトは `PROMPTS["entity_extraction"]` を使用し、出力をレコード区切り/タプル区切りでパースします。
  - エンティティ名・タイプは大文字化、`source_id` にはチャンクキー（chunk のID）が格納されます。
  - `relationship_strength` が数値でない場合、重みは既定で `1.0` になります。

- **マージルールの正確な挙動**
  - ノード: `entity_type` は多数決、`description`/`source_id` は重複排除の上で `<SEP>` 連結（`GRAPH_FIELD_SEP`）。
  - エッジ: `weight` は合算、`description`/`keywords`/`source_id` は重複排除の上で `<SEP>` 連結。
  - 欠損ノードはエッジUpsert前に自動作成されますが、その際の `entity_type` は実装上 `"UNKNOWN"`（二重引用を含む文字列）として保存されます。

- **Neo4j のノード作成タイミング**
  - エッジUpsertのCypher自体は `MATCH`→`MERGE (source)-[r:DIRECTED]->(target)` ですが、欠損ノードの自動生成は抽出層（Python側）で `upsert_node` を先に呼び出すことで担保しています。

- **VDB への保存内容**
  - `entity_vdb` にはエンティティ名と説明を連結した `content` を保存、`entity_name_vdb` には名前のみ、`relationships_vdb` には `keywords + src + tgt + description` を連結した `content` を保存します。
  - 実装では `entity_vdb.upsert` が2回呼ばれており（ほぼ同内容）、機能上は問題ないものの冗長です（最終的な内容は2回目が反映）。

- **チャンクおよびドキュメントの管理**
  - チャンクは `chunks_vdb` と `text_chunks`（KV）に保存、元ドキュメントは `full_docs`（KV）に保存され、`DocStatus` により処理状態を管理します。

## 軽微な是正提案

- **冗長な VDB 書き込みの整理**: `entity_vdb` への二重 upsert を1回に統一。
  - 影響: 機能は不変、処理の明瞭化と微小なコスト削減。

- **`UNKNOWN` 表記の統一**: `entity_type` の `"UNKNOWN"` を `UNKNOWN` に統一し、表記揺れと二重引用の混在を解消。
  - 影響: 既存比較ロジックの単純化（現状は `strip('"')` で吸収済み）。

## 追加実装（出典情報の返却オプション）

- `QueryParam` に `include_provenance: bool`（既定 False）を追加。
- Mini モードの `minirag_query` を拡張し、`include_provenance=True` の場合に、回答と併せて構造化された出典情報（entities/chunks）を返却可能にしました。
- 返却仕様:
  - `MiniRAG.query(...)` は常にタプル `(first, source)` を返します。`first` の中身は以下の通りです。
    - `include_provenance=False`（既定）: 文字列のみ（回答 or コンテキスト）
    - `include_provenance=True` の通常回答時: `{ "answer": <str>, "provenance": { "entities": [...], "chunks": [...] } }`
    - `include_provenance=True` かつ `only_need_context=True` の時: `{ "context": <str>, "provenance": {...} }`
  - `source` は従来通り、使用したソーステキスト（チャンク内容文字列）のリスト。

- 影響範囲
  - 他モード（light/global, naive）には変更なし。
  - 型的には `MiniRAG.query` の第一戻り値が str もしくは dict になり得る（`include_provenance=True` のとき）。既存利用コードが文字列前提なら `include_provenance=False` のまま使ってください。

### provenance 内容
- `entities`: `[{ entity_name, score, description }]`
- `chunks`: `[{ chunk_id, full_doc_id, chunk_order_index, tokens, content }]`

### 使い方（例）
```python
from minirag.base import QueryParam

param = QueryParam(mode="mini", include_provenance=True)
result, _ = rag.query("質問文", param=param)
# result == {"answer": "...", "provenance": {"entities": [...], "chunks": [...]}}

# コンテキストのみ欲しい場合
param = QueryParam(mode="mini", include_provenance=True, only_need_context=True)
ctx, _ = rag.query("質問文", param=param)
# ctx == {"context": "...", "provenance": {...}}
```

## 参考（主要ポイントの所在）

- チャンク化: `minirag/operate.py` `chunking_by_token_size`
- 抽出・マージ・Upsert: `minirag/operate.py` `extract_entities`, `_handle_single_entity_extraction`, `_handle_single_relationship_extraction`, `_merge_nodes_then_upsert`, `_merge_edges_then_upsert`
- ストレージ層（NetworkX/Neo4j）: `minirag/kg/networkx_impl.py`, `minirag/kg/neo4j_impl.py`
- プロンプト定義: `minirag/prompt.py`
- パイプラインとVDB/KV初期化: `minirag/minirag.py`
- 出典情報の生成/返却箇所: `minirag/operate.py` `_build_mini_query_context`, `minirag_query`


