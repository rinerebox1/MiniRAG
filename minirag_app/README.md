# MiniRAG: 極めてシンプルな検索強化生成に向けて

![MiniRAG](https://files.mdnice.com/user/87760/ff711e74-c382-4432-bec2-e6f2aa787df1.jpg)

- [done]PostgreSQL + pgvector + Apache AGE の Docker はできたので、次は MiniRAG を Docker で構築してみる。
- [done]PostgreSQL16_pgvector(HNSW)_ApacheAGE_Docker に作っちゃったけど、最後には1つの Docker にまとめたい。


chmod +x scripts/build.sh
chmod +x scripts/start.sh
chmod +x scripts/stop.sh

http://localhost:8165/tree?


作成したデータベースは「MiniRAG/data/postgres」に保存される。


◯フィルター機能について
フィルター機能を入れなければめちゃくちゃ簡単だが、フィルター機能いれるととてつもなく難しく、不安定になった気がする。
せっかく入れたので一旦フィルター機能は入れたままにする。
あと、デバッグ表示するためだけにコードを改良した部分もあり。不用意に複雑化した。
→メモリームに入れるためにフィルター機能を入れた。




◯ソースコードを変更したらこれでいい（データベースをクリーンアップする）
sudo scripts/start.sh cleanup

◯ <tool_call> がめちゃくちゃ出るのは、多分 LLM のせい。今は有料で Qwen3 を使っている。



[LOCAL] 🔍 Metadata filter {'category': 'weather'} applied at KG stage: 5 -> 2 chunks matched
[GLOBAL] 🔍 Metadata filter {'category': 'weather'} applied at KG stage: 0 -> 0 chunks matched

→ナレッジグラフはローカルクエリとグローバルクエリの2回実行される。コンテキストとして渡されるのは両方の合算。


■ KG(ナレッジグラフ) ↔ VDB(ベクトルDB) の整合性を保つためのデリート処理は大事。(あまりないケースだと思うけど)重複するドキュメントを登録するときに一回デリートする処理が必要っぽいので、こういう実装が必要になってきた。デリート処理は非効率な気がするけど、こうしないとうまくいかなかった。

● PGVectorStorage
　・delete_entity() / delete_relation() を追加し、エンティティ名に基づくベクタ削除を実装
　　（entities / entities_name テーブル、relationships テーブルに対応）
● PGGraphStorage
　・delete_node() を実装。DETACH DELETE によりノードと関連エッジを AGE から一括削除
● MiniRAG.adelete_by_entity
　・entity_name_vdb.delete_entity() も呼び出し、全 VDB で完全同期を確保
これで
・ノード削除 → エッジも自動削除（DETACH DELETE）
・削除されたエンティティ／エッジに関連する VDB レコードも確実に削除
という KG ↔ VDB の整合性が取れる削除フローが完成しました。




問題が明確になりました！デバッグ情報から以下のことがわかります：

1. **データベースには5つのチャンクが存在**
2. **しかし、chunk 0と1のmetadataがNone**
3. **chunk 2のmetadataは`{"year": 2023, "country": "Japan", "category": "geography"}`**
4. **`category: "weather"`のデータが存在しない**

つまり、テストで挿入したはずの`{"category": "weather", "city": "Tokyo", "year": 2024}`のメタデータが正しく保存されていないということです。

問題の原因を特定するために、チャンクのupsert処理を詳しく調べましょう。
Ran tool
問題が見えました！`apipeline_process_enqueue_documents`の処理を見ると、チャンクにメタデータが追加される箇所は以下の通りです：

```python
chunks = {
    compute_mdhash_id(dp["content"], prefix="chunk-"): {
        **dp,
        "full_doc_id": doc_id,
        "metadata": status_doc.metadata or {},
    }
    for dp in self.chunking_func(...)
}
```

この`status_doc.metadata`は、ドキュメントのステータスから取得されていますが、デバッグ結果を見ると、一部のチャンクにメタデータが設定されていません。

これは、既存のドキュメントが再処理される際に、新しいメタデータが反映されていない可能性があります。

`doc_status.filter_keys`で既存のドキュメントが除外されている可能性があります。確認してみましょう：
Read file: minirag_app/minirag/kg/postgres_impl.py
問題が判明しました！`filter_keys`は既存のドキュメントIDを除外するため、同じドキュメントを異なるメタデータで再挿入しようとすると、新しいメタデータが反映されません。

この問題を解決するには、以下のいずれかのアプローチが必要です：

1. **既存ドキュメントを強制更新する**
2. **異なるIDを使用する**

まず、テスト用に一時的に`filter_keys`をバイパスして、常に新しいドキュメントとして扱うように修正してみましょう：
さらに、メタデータが正しく設定されているかをデバッグするために、ドキュメント挿入時のログも追加しましょう：
さらに、チャンク処理時にもメタデータのログを追加しましょう：
これで修正が完了しました。主な変更点は：

1. **`filter_keys`のバイパス**：既存ドキュメントをフィルタリングしないで、強制的に更新する
2. **詳細なデバッグログ**：
   - ドキュメント挿入時のメタデータ
   - チャンク処理時のメタデータ
   - 作成されたチャンク数

これにより、メタデータが正しく流れているかどうかを確認できます。

**推測される問題の原因**：
- テストで同じドキュメントIDを使用している場合、`filter_keys`によって既存ドキュメントが除外され、新しいメタデータが反映されない
- 結果として、古いチャンク（メタデータなし）がデータベースに残っている

**解決策**：
1. **短期的**：テスト時に毎回異なるIDを使用する、または上記の修正でfilter_keysをバイパス
2. **長期的**：既存ドキュメントのメタデータ更新機能を実装する

ユーザーのテストを再実行すると、今度はメタデータが正しく保存され、フィルタリングが動作するはずです。


---
★動いた。厳密なフィルタリングにはならなかったけど、最終的にはメタデータでフィルターできているのでこれでOK(最初にフィルターなしでガバっと取ってくる)

■ この変更は「厳密な絞り込み」から「意味的な関連性を優先した上での絞り込み」へと、検索の哲学を転換させた。これによって、メタデータはチャンク側にさえしっかり付与されていれば良いため、エンティティやリレーションシップのベクトルに、必ずしもチャンクレベルの全メタデータ（例：ファイル名、作成部署など）を埋め込む必要がなくなります。

- 修正前: Filter -> Search。最初に候補を大幅に絞り込んでからベクトル検索を行うため、高速です。

- 修正後: Search -> Filter。まず広範囲にベクトル検索を行い、多くの候補（エンティティ/リレーションシップ）とそれに関連するチャンクを取得してから、アプリケーション側でフィルタリングします。
- まずメタデータに関係なく「製品ロードマップ」というエンティティを特定します。そのエンティティが参照しているチャンクの中に、セールス部門が作成したドキュメント（"department": "Sales"）があれば、それを最終的な結果として返すことができます。これにより、よりユーザーの意図に沿った結果を見つけられる可能性が高まります。
- この修正後の挙動は、一度その範囲を無視して検索するため、ユーザーの直感に反する可能性があります。「なぜ営業部門でフィルタしたのに、開発部門のエンティティまで検索しているんだ？」と内部動作を知ると混乱するかもしれません。厳密性が求められるユースケース（例：法務監査）では、この挙動は問題となる可能性があります。

解説：ライト／ミニは
① relationships（または entities）を検索
② 関連ノード／エッジをたどって チャンク ID を取得
③ text_chunks_db からチャンク本文を読む
という 2 段階構成です。

↓

はい、続けて`_build_global_query_context`を修正します。
お待たせいたしました。`hybrid_query`におけるメタデータフィルタリングの問題を修正しました。

### 修正内容

1.  **`_build_local_query_context` 関数の修正**:
    *   `entities_vdb.query` 呼び出しから `metadata_filter` を削除しました。
    *   代わりに、`_find_most_related_text_unit_from_entities` でチャンクを取得した**後**に、`query_param.metadata_filter` を使ってチャンクをフィルタリングするロジックを追加しました。

2.  **`_build_global_query_context` 関数の修正**:
    *   `relationships_vdb.query` 呼び出しから `metadata_filter` を削除しました。
    *   こちらも同様に、`_find_related_text_unit_from_relationships` でチャンクを取得した**後**に、`query_param.metadata_filter` でチャンクをフィルタリングするロジックを追加しました。

### 修正による効果

この修正により、ライトモード (`hybrid_query`) の動作は以下のようになります。

1.  まず、クエリキーワードに基づいて、メタデータに関係なく関連性の高いエンティティとリレーションシップを幅広く検索します。
2.  次に、それらのエンティティとリレーションシップが参照しているチャンクをすべて取得します。
3.  最後に、取得したチャンクのリストに対して、ユーザーが指定した `metadata_filter` を適用し、最終的なソースを絞り込みます。

これにより、エンティティのメタデータに依存することなく、ユーザーの意図通りに**チャンクのメタデータ**でフィルタリングが行われ、期待されるソースが正しく返されるようになります。

再度、動作をご確認いただけますでしょうか。



----

print(os.getenv("DATABASE_URL"))
# postgresql://postgres_user:postgres_pass@postgres:5432/my_database


### ハイブリッドクエリのテスト
```
docker exec -it postgres16_age_pgvector_container psql -U postgres_user -d my_database
```
SELECT p.id, p.name, p.embedding FROM public.products AS p JOIN cypher('my_minirag_graph', $$ MATCH (u:User {name: 'Alice'})-[:LIKES]->(prod:Product) RETURN prod.product_id $$) AS liked(product_id agtype) ON p.id = (liked.product_id)::INTEGER ORDER BY p.embedding <=> '[0.1, 0.1, 0.2]';


---

already_weights.append(already_edge["weight"])
'NoneType' object is not subscriptable
というエラーが出る問題。

はい。今回の修正後は、通常フローで挿入・更新されるエッジには必ず `weight` が数値で入る設計になっています。

根拠
1. 抽出段階 (`_handle_single_relationship_extraction`)  
   • LLM から返るレコード末尾が数値ならその値、無い場合でも `1.0` をデフォルトにして  
   ```python
   weight = float(record_attributes[-1]) if is_float_regex(... ) else 1.0
   ```
   を設定しています。  
2. マージ段階 (`_merge_edges_then_upsert`)  
   • `weight = sum([dp["weight"] for dp in edges_data] + already_weights)` で必ず数値を生成。  
   • もし既存エッジに `weight` が無くても `already_edge.get("weight", 0.0)` で 0 を補完し、  
     新しい `weight` を含む dict を `upsert_edge` に渡します。  
3. DB 反映 (`upsert_edge`)  
   • `SET r += {weight: ..., ...}` の形で PG/AGE に書き込むため、`weight` キーが空になることはありません。  

注意点
• パイプライン外で手動挿入する場合は `weight` を必ず指定してください。  
• 旧データを移行せず残すと “プロパティ無しエッジ” が再発する可能性がありますが、  
  今回の再ビルドで DB をまっさらにするため問題ありません。

したがって、今後生成されるエッジには最低でも `weight = 1.0` が入ります。




## MiniRAG の Docker

docker compose build --no-cache

docker compose up -d

docker compose down -v


http://localhost:8165/tree?


## MiniRAG クエリモード解説

MiniRAGは、検索拡張生成（RAG）のプロセスを最適化するために、複数のクエリモードを提供しています。各モードは、速度、コスト、および回答の品質のバランスが異なるため、ユースケースに応じて最適なものを選択できます。

### 1. NAIVEモード

**NAIVEモード**は、最もシンプルで直接的なRAGの手法です。

#### 仕組み

1.  ユーザーのクエリに基づいて、ベクトルデータベース（`chunks_vdb`）から関連性の高いテキストチャンクを検索します。
2.  取得したテキストチャンクをコンテキストとして大規模言語モデル（LLM）に提供します。
3.  LLMは、そのコンテキストに基づいて回答を生成します。

#### 特徴

*   **実装が容易**: 知識グラフなどの複雑なデータ構造を必要とせず、ベクトル検索のみに依存します。
*   **高速**: クエリ処理が単純なため、迅速に回答を得られます。
*   **低コスト**: 複雑な処理が不要なため、計算リソースの消費が少なくて済みます。

#### ユースケース

*   迅速な回答生成が求められる場合。
*   ドキュメントが構造化されておらず、エンティティや関係性の抽出が困難な場合。
*   シンプルなQ&Aシステム。

---

### 2. LIGHTモード

**LIGHTモード**は、`NAIVE`モードと`MINI`モードの中間に位置し、知識グラフを活用してより高品質なコンテキストを生成します。

#### 仕組み

1.  LLMを使用して、ユーザークエリから**高レベル**（概念、トピック）と**低レベル**（具体的なエンティティ名）のキーワードを抽出します。
2.  **高レベルキーワード**は、関係性ベクトルDB（`relationships_vdb`）を検索し、関連する関係性を特定します。
3.  **低レベルキーワード**は、エンティティベクトルDB（`entities_vdb`）を検索し、関連するエンティティを特定します。
4.  これらのエンティティと関係性から、知識グラフ上で関連するテキストチャンクや近傍エンティティを収集し、コンテキストを構築します。
5.  構築されたコンテキストをLLMに提供し、回答を生成します。

#### 特徴

*   **バランスの取れた性能**: `NAIVE`モードよりも高品質な回答を生成しつつ、`MINI`モードよりも高速に動作します。
*   **知識グラフの活用**: エンティティと関係性の両方を考慮することで、よりリッチなコンテキストを生成します。
*   **ハイブリッド検索**: キーワードのレベルに応じて異なる検索戦略を用いることで、効率的に情報を収集します。

#### ユースケース

*   ある程度の回答品質が求められ、かつリアルタイム性も重要な場合。
*   構造化されたデータと非構造化データが混在するドキュメントセット。
*   より複雑な質問に答える必要がある場合。

---

### 3. MINIモード

**MINIモード**は、最も高度で包括的なRAGの手法であり、知識グラフを最大限に活用して、深い推論に基づいた回答を生成します。
ナレッジグラフとベクトルデータベースを組み合わせた**ハイブリッドなRAG（Retrieval-Augmented Generation）**を実行します。「グラフ推論で重要とされたチャンク」と「ベクトル検索で重要とされたチャンク」と「両方で重要とされたチャンクは特に高く評価」のスコアを合算し、最終的にLLMに渡すための最も重要なチャンクIDのリスト（final_chunk_id）を決定します。

#### 仕組み

1.  LLMを使用して、ユーザークエリから**回答の期待される型**（例：「人名」「組織名」）と**クエリ内のエンティティ**を抽出します。
2.  抽出されたエンティティを基点に、知識グラフ上で**kホップ**内（このコードでは2ホップ）の近傍ノードを探索し、推論パスの候補を洗い出します。
3.  クエリとの関連性や、期待される回答の型に合致するノードへのパスを評価し、スコアリングします。
4.  最もスコアの高い推論パスに関連するエンティティ、関係性、テキストチャンクを収集し、詳細なコンテキストを構築します。
5.  構築されたコンテキストをLLMに提供し、回答を生成します。

#### 特徴

*   **最高の回答品質**: 知識グラフ上の多段階の推論（Multi-hop reasoning）により、直接的な情報だけでなく、間接的な関係性も考慮した深い回答を生成できます。
*   **詳細なコンテキスト**: 最も関連性の高い情報に絞り込むため、ノイズの少ない、質の高いコンテキストをLLMに提供できます。
*   **複雑なクエリへの対応**: 「AとBの関係は？」といった、複数のエンティティや関係性をまたぐ複雑な質問に効果的です。

#### ユースケース

*   最高の回答精度が求められる場合。
*   金融分析、科学研究、医療診断など、専門的な知識を要する分野。
*   複雑な因果関係や時系列の分析が必要な場合。

---

### まとめ

| モード | 特徴 | 長所 | 短所 |
| --- | --- | --- | --- |
| **NAIVE** | ベクトル検索のみ | 高速、低コスト、シンプル | 回答の質が低い可能性がある |
| **LIGHT** | エンティティと関係性のハイブリッド検索 | バランスの取れた性能 | NAIVEより遅く、MINIより質が低い |
| **MINI** | 知識グラフ上での多段階推論 | 最高の回答品質、複雑なクエリに対応 | 最も遅く、計算コストが高い |


## あるクエリでは、hybrid_query関数(LIGHTモード)ではうまくコンテキストが取得できるのに、minirag_query関数(MINIモード)ではコンテキストが取得できない。これはあり得る。それはエンティティをうまく抽出できない場合。なので、両方使った方が良さそう。

はい、その現象は十分に起こり得ます。`hybrid_query`ではうまくコンテキストが取得でき、`minirag_query`ではできない場合、その原因は両者の**コンテキスト構築戦略の根本的な違い**に起因していると考えられます。

以下に、考えられる主な原因をいくつか挙げます。

### 1. キーワード抽出の戦略の違い

これが最も大きな原因である可能性が高いです。

*   **`hybrid_query`**:
    *   LLMに「**高レベル（抽象的）なキーワード**」と「**低レベル（具体的）なキーワード**」の両方を抽出させます。
    *   **低レベルキーワード**には、クエリ内の具体的なエンティティ名や固有名詞が含まれやすいです。
    *   この低レベルキーワードを使って`_build_local_query_context`が実行され、エンティティのベクトルDB（`entities_vdb`）を直接検索します。
    *   **クエリに具体的なエンティティ名が含まれていれば、この時点でヒットし、コンテキスト構築が開始されやすいです。**

*   **`minirag_query`**:
    *   LLMに「**`answer_type_keywords`（回答の型）**」と「**`entities_from_query`（クエリ内のエンティティ）**」を抽出させます。
    *   問題は、**LLMがクエリからエンティティをうまく抽出できなかった場合**です。例えば、マイナーな固有名詞や、LLMがエンティティとして認識しにくい単語だった場合、`entities_from_query`が空になる可能性があります。
    *   `entities_from_query`が空だと、グラフをたどり始める**起点（エントリーポイント）**を失います。その結果、後続のグラフ推論が全く機能せず、コンテキストが空になってしまいます。

**シナリオ例**:
クエリが「〇〇社の××という製品の仕様は？」だったとします。

*   `hybrid_query`のLLMは「低レベルキーワード」として「〇〇社」「××」を抽出し、それを元に`entities_vdb`を検索してコンテキストを構築できます。
*   `minirag_query`のLLMが「〇〇社」や「××」をエンティティとして認識できず、`entities_from_query`が空になった場合、グラフ探索の始点がなくなり、処理が失敗します。

### 2. コンテキスト構築の依存関係の違い

*   **`hybrid_query`**:
    *   `_build_local_query_context`（低レベル）と`_build_global_query_context`（高レベル）は、それぞれ独立してコンテキストを構築しようと試みます。
    *   片方（例: 低レベルキーワードでの検索）が成功すれば、少なくともその部分のコンテキストは得られます。両方が失敗しない限り、何かしらのコンテキストが返る可能性が高いです。**フォールバックが効きやすい構造**と言えます。

*   **`minirag_query`**:
    *   `_build_mini_query_context`の処理は、**連鎖的（シーケンシャル）**です。
    *   `entities_from_query`の抽出 → グラフの近傍探索 → パスのスコアリング → チャンクへのマッピング、という一連の流れが一つでも失敗すると、後続の処理がすべて意味をなさなくなり、最終的にコンテキストが空になります。
    *   特に、**最初のエンティティ抽出（グラフへの入り口）に強く依存している**ため、ここがボトルネックになりやすいです。

### 3. プロンプトの性質の違い

*   **`hybrid_query`のプロンプト (`keywords_extraction`)**:
    *   比較的シンプルに、クエリから抽象的・具体的なキーワードを抜き出すことを指示しています。タスクが単純なため、LLMは安定して何らかのキーワードを抽出しやすいです。

*   **`minirag_query`のプロンプト (`minirag_query2kwd`)**:
    *   「回答の型」と「クエリ内のエンティティ」を特定するという、より複雑で制約の厳しいタスクをLLMに課しています。
    *   提供されている`TYPE_POOL`（エンティティタイプのリスト）に合致する型が見つからなかったり、クエリの表現が曖昧でエンティティを特定しきれなかったりすると、LLMが空の結果を返してしまうリスクが相対的に高いです。

### 4. 対象とするクエリの種類の違い（設計思想）

*   **`hybrid_query`**:
    *   具体的なエンティティに関する質問（**Local Query**）と、抽象的な概念や関係性に関する質問（**Global Query**）の両方に対応できるよう設計されています。キーワードベースで幅広く情報を拾うため、**網羅的**です。

*   **`minirag_query`**:
    *   グラフの**推論能力**を最大限に活用することに特化しています。「AとBの関係は？」「Cという特徴を持つDは？」といった、エンティティ間の繋がりをたどる必要がある**複雑な質問**で真価を発揮するように設計されています。
    *   逆に、単純な事実確認のようなクエリで、エンティティ抽出がうまくいかないと、その能力を発揮できずに失敗することがあります。

---

### 結論と解決策の方向性

**原因の要約**:
「`hybrid_query`では成功し、`minirag_query`では失敗する」という現象の最も可能性の高い原因は、**`minirag_query`が最初のステップである「クエリからのエンティティ抽出」に失敗し、グラフ推論の起点を見つけられなかったため**です。`hybrid_query`は、より単純なキーワード抽出とフォールバックが効きやすい構造のため、その問題を回避できています。

**考えられる対策**:

1.  **プロンプトの改善**: `minirag_query`で使われる`PROMPTS["minirag_query2kwd"]`を改善し、LLMがより頑健にエンティティを抽出できるようにする（例: few-shot learningで例を示す、指示をより具体的にするなど）。
2.  **フォールバック機構の導入**: `minirag_query`でエンティティが抽出できなかった場合に、`hybrid_query`の低レベルキーワード抽出のような、よりシンプルな方法でグラフの起点を探す処理を追加する。
3.  **クエリの書き方の工夫**: `minirag_query`を使う際に、ユーザーがエンティティを明確に（例: `「」`で括るなど）記述するようにガイドする。

もし特定のクエリでこの問題が発生している場合、そのクエリを例に、両方の関数がどのような中間結果（キーワードやエンティティ）を抽出しているかデバッグしてみると、原因がより明確に特定できるでしょう。

---

コードリポジトリ: **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation**
<br />

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />
</div>

<a href='https://arxiv.org/abs/2501.06713'><img src='https://img.shields.io/badge/arXiv-2501.06713-b31b1b'>


## 🌍 READMEの翻訳

[English](./README.md) | [中文](./README_CN.md)

## 🎉 News
- [x] [2025.02.27]🎯📢`pip install minirag-hku`を使用して私たちのコードを実行できるようになりました！
- [x] [2025.02.14]🎯📢MiniRAGがNeo4j、PostgreSQL、TiDBなど10以上の異種グラフデータベースをサポートするようになりました。バレンタインデーおめでとう！🌹🌹🌹
- [x] [2025.02.05]🎯📢私たちのチームは、非常に長いコンテキストの動画を理解するVideoRAGをリリースしました。
- [x] [2025.02.01]🎯📢MiniRAGがAPI&Dockerデプロイメントをサポートするようになりました。詳細はこちらをご覧ください。

## TLDR
MiniRAGは、異種グラフインデックスと軽量なトポロジー強化検索を通じて、小さなモデルでも優れたRAGパフォーマンスを実現する極めてシンプルな検索強化生成フレームワークです。

## 概要
効率的で軽量な検索強化生成（RAG）システムの需要が高まる中、既存のRAGフレームワークに小型言語モデル（SLM）を導入する際の重大な課題が浮き彫りになっています。現在のアプローチは、SLMの限られた意味理解とテキスト処理能力のために深刻な性能低下に直面しており、リソースが限られたシナリオでの広範な採用に障害をもたらしています。これらの根本的な制限に対処するために、私たちは極めてシンプルで効率的な新しいRAGシステムである**MiniRAG**を提案します。**MiniRAG**は、2つの重要な技術革新を導入しています：（1）テキストチャンクと名前付きエンティティを統一構造に組み合わせる意味認識異種グラフインデックスメカニズム、これにより複雑な意味理解への依存を減らします。（2）高度な言語能力を必要とせずにグラフ構造を活用して効率的な知識発見を実現する軽量なトポロジー強化検索アプローチ。私たちの広範な実験は、**MiniRAG**がSLMを使用してもLLMベースの方法と同等の性能を達成しながら、ストレージスペースの25％しか必要としないことを示しています。さらに、複雑なクエリを持つ現実的なオンデバイスシナリオで軽量RAGシステムを評価するための包括的なベンチマークデータセットLiHua-Worldも提供します。

## MiniRAGフレームワーク

![MiniRAG](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

MiniRAGは、異種グラフインデックスと軽量なグラフベースの知識検索という主要なコンポーネントに基づいて構築された簡素化されたワークフローを採用しています。このアーキテクチャは、オンデバイスRAGシステムが直面する独自の課題に対処し、効率と効果の両方を最適化します。


## インストール

* ソースからインストール（推奨）

```bash
cd MiniRAG
pip install -e .
```
* PyPIからインストール（私たちのコードは[LightRAG](https://github.com/HKUDS/LightRAG)に基づいているため、直接インストールできます）

```bash
pip install lightrag-hku
```

## クイックスタート
* すべてのコードは`./reproduce`にあります。
* 必要なデータセットをダウンロードします。
* データセットを`./dataset`ディレクトリに配置します。
* 注：LiHua-Worldデータセットは`./dataset/LiHua-World/data/`に`LiHuaWorld.zip`として既に配置されています。他のデータセットを使用したい場合は、`./dataset/xxx`に配置できます。


次に、以下のbashコマンドを使用してデータセットをインデックスします：
```bash
python ./reproduce/Step_0_index.py
python ./reproduce/Step_1_QA.py
```

または、`./main.py`のコードを使用してMiniRAGを初期化します。


### 全体のパフォーマンステーブル
| モデル | NaiveRAG | | GraphRAG | | LightRAG | | **MiniRAG** | |
|-------|----------|----------|-----------|----------|-----------|----------|----------|----------|
| | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ | acc↑ | err↓ |
| LiHua-World | | | | | | | | |
| Phi-3.5-mini-instruct | 41.22% | 23.20% | / | / | 39.81% | 25.39% | **53.29%** | 23.35% |
| GLM-Edge-1.5B-Chat | 42.79% | 24.76% | / | / | 35.74% | 25.86% | **52.51%** | 25.71% |
| Qwen2.5-3B-Instruct | 43.73% | 24.14% | / | / | 39.18% | 28.68% | **48.75%** | 26.02% |
| MiniCPM3-4B | 43.42% | 17.08% | / | / | 35.42% | 21.94% | **51.25%** | 21.79% |
| gpt-4o-mini | 46.55% | 19.12% | 35.27% | 37.77% | **56.90%** | 20.85% | 54.08% | 19.44% |
| MultiHop-RAG | | | | | | | | |
| Phi-3.5-mini-instruct | 42.72% | 31.34% | / | / | 27.03% | 11.78% | **49.96%** | 28.44% |
| GLM-Edge-1.5B-Chat | 44.44% | 24.26% | / | / | / | / | **51.41%** | 23.44% |
| Qwen2.5-3B-Instruct | 39.48% | 31.69% | / | / | 21.91% | 13.73% | **48.55%** | 33.10% |
| MiniCPM3-4B | 39.24% | 31.42% | / | / | 19.48% | 10.41% | **47.77%** | 26.88% |
| gpt-4o-mini | 53.60% | 27.19% | 60.92% | 16.86% | 64.91% | 19.37% | **68.43%** | 19.41% |


表中、/はその方法が効果的な応答を生成するのに苦労していることを意味します。

## 再現
すべてのコードは`./reproduce`ディレクトリにあります。

## コード構造

```python
├── dataset
│   └── LiHua-World
│       ├── README.md
│       ├── README_CN.md
│       ├── data
│       │   ├── LiHuaWorld.zip
│       └── qa
│           ├── query_set.csv
│           └── query_set.json
├── minirag
│   ├── kg
│   │   ├── __init__.py
│   │   ├── neo4j_impl.py
│   │   └── oracle_impl.py
│   ├── __init__.py
│   ├── base.py
│   ├── llm.py
│   ├── minirag.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.py
├── reproduce
│   ├── Step_0_index.py
│   └── Step_1_QA.py
├── LICENSE
├── main.py
├── README.md
├── README_CN.md
├── requirements.txt
├── setup.py
```

## データセット: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-Worldは、仮想ユーザーLiHuaの1年間のチャット記録を含む、オンデバイスRAGシナリオ専用に設計されたデータセットです。このデータセットには、シングルホップ、マルチホップ、およびサマリーの3種類の質問が含まれており、各質問には手動で注釈が付けられた回答とサポート文書がペアになっています。詳細については、[LiHua-WorldデータセットのREADME](./dataset/LiHua-World/README.md)を参照してください。


## Star History

<a href="https://star-history.com/#HKUDS/MiniRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/MiniRAG&type=Date" />
 </picture>
</a>

## Contribution

MiniRAGプロジェクトのすべての貢献者に感謝します！

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>

## 謝辞
私たちのフレームワークとコードリポジトリの基礎となる関連作業については、[nano-graphrag](https://github.com/gusye1234/nano-graphrag)および[LightRAG](https://github.com/HKUDS/LightRAG)を参照してください。素晴らしい仕事に感謝します。

## 🌟引用

```python
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

**私たちの仕事に興味を持っていただき、ありがとうございます！**