## 前提条件

MiniRAGでPostgreSQLを使用する前に、以下の前提条件が満たされていることを確認してください。

### PostgreSQLサーバー
*   **バージョン**: MiniRAGは一般的に最新のPostgreSQLバージョンと互換性があります。ソース内で特定の最小バージョンは明示されていませんが、パーティショニングやその他の有益なパフォーマンス向上機能を含む幅広い機能サポートのために、PostgreSQL 12+を推奨します。Apache AGEおよびpgvectorについては、それぞれのドキュメントでPostgreSQLバージョンの互換性を参照してください。
*   **pgvector拡張機能**: ベクトル類似性検索機能（`PGVectorStorage`で使用）のためには、PostgreSQLデータベースに`pgvector`拡張機能がインストールされている必要があります。通常、データベースサーバーで拡張機能が利用可能な場合は、`CREATE EXTENSION IF NOT EXISTS vector;`を使用してインストールできます。
*   **Apache AGE拡張機能**: グラフデータベース機能（`PGGraphStorage`で使用）のためには、Apache AGE (A Graph Extension) がPostgreSQLデータベースにインストールされ、設定されている必要があります。お使いのPostgreSQLバージョンに対応する公式のApache AGEインストールガイドに従ってください。

### Pythonの依存関係
MiniRAG環境に以下のPythonライブラリがインストールされていることを確認してください。MiniRAGは`asyncpg`が見つからない場合にインストールを試みますが、依存関係を明示的に管理することが推奨されます。
*   `asyncpg`: PostgreSQLとの非同期対話用。
*   `psycopg-pool`: コネクションプーリング用（テストファイルで言及されており、堅牢なアプリケーションに適しています）。
*   `psycopg[binary,pool]`: バイナリおよびプーリングサポートを備えた代替PostgreSQLアダプタ（テストファイルで言及されています）。

通常、これらはpipを使用してインストールできます:
```bash
pip install asyncpg psycopg-pool "psycopg[binary,pool]"
```

## 設定

MiniRAGをPostgreSQLインスタンスに接続するには、接続パラメータを提供する必要があります。これは通常、データベース対話を必要とするMiniRAGコンポーネントを初期化する際に行われます。

PostgreSQL接続を管理する主要なクラスは`PostgreSQLDB`です。以下のキーを持つ設定辞書を期待します:

*   `host` (str): PostgreSQLサーバーのホスト名またはIPアドレス。デフォルト: `"localhost"`。
*   `port` (int): PostgreSQLがリッスンしているポート番号。デフォルト: `5432`。
*   `user` (str): PostgreSQLのユーザー名。**必須。**
*   `password` (str): 指定されたユーザーのパスワード。**必須。**
*   `database` (str): 接続するPostgreSQLデータベースの名前。**必須。**
*   `workspace` (str): 異なるプロジェクトやインスタンスのデータを分離するためのMiniRAGテーブル内の名前空間。デフォルト: `"default"`。これにより、データが衝突することなく、複数のMiniRAGセットアップで同じデータベースを使用できます。MiniRAGによって作成されるすべてのテーブルエントリは、このワークスペース識別子に関連付けられます。

### 設定例

Pythonでの設定辞書の構成例です:

```python
postgres_config = {
    "host": "your_postgres_host",
    "port": 5432,
    "user": "your_postgres_user",
    "password": "your_postgres_password",
    "database": "minirag_db",
    "workspace": "my_project_space"
}

# この設定は、MiniRAGコンポーネントに渡されます。例:
# from minirag.kg.postgres_impl import PostgreSQLDB
# db_instance = PostgreSQLDB(config=postgres_config)
# (コンポーネント初期化の詳細は「MiniRAGでの使用法」で説明します)
```

**注意:** 指定された`user`が、対象の`database`に対する必要な権限（CREATE、SELECT、INSERT、UPDATE、DELETE）を持っていること、および拡張機能がまだインストールされていない場合にそれらを作成するための権限を持っていることを確認してください（ただし、多くの場合、拡張機能はスーパーユーザーによって事前にインストールされている方が良いです）。

## 初期化とテーブル作成

MiniRAGがPostgreSQLへの接続を初期化する際（例: `PostgreSQLDB.initdb()`が呼び出された時）、必要なテーブルの存在を自動的に確認します。テーブルが見つからない場合、MiniRAGはそれらの作成を試みます。

これらのテーブルのDDL（データ定義言語）は、`minirag/kg/postgres_impl.py`内に内部的に定義されています。MiniRAGによって作成されるテーブルには以下が含まれます:

*   `LIGHTRAG_DOC_FULL`: 取り込まれたドキュメントの完全なコンテンツを保存します。
*   `LIGHTRAG_DOC_CHUNKS`: ドキュメントから処理されたテキストチャンクを保存します。ベクトル埋め込みも含む場合があります（`pgvector`を使用している場合）。
*   `LIGHTRAG_VDB_ENTITY`: セマンティック検索に使用されるエンティティのベクトル埋め込みを保存します。
*   `LIGHTRAG_VDB_RELATION`: エンティティ間のリレーションシップのベクトル埋め込みを保存します。
*   `LIGHTRAG_LLM_CACHE`: 言語モデル（LLM）からの応答をキャッシュし、冗長なAPI呼び出しを回避します。
*   `LIGHTRAG_DOC_STATUS`: MiniRAGに取り込まれたドキュメントの処理ステータスを追跡します。

すべてのテーブルは、デフォルトでパブリックスキーマ内に作成され（PostgreSQLユーザーまたは検索パスが異なる設定になっていない限り）、接続設定で指定されたデータを分離するための`workspace`列を含みます。

MiniRAGはテーブル作成を処理しますが、設定されたPostgreSQLの`user`が指定された`database`にテーブルを作成するための必要な権限を持っていることが重要です。

## Apache AGE (A Graph Extension) のセットアップ

MiniRAGは、`PGGraphStorage`クラスに実装されているグラフストレージおよびクエリ機能のためにApache AGEを利用します。

### 前提条件
*   **Apache AGEのインストール**: 前提条件で述べたように、Apache AGEがPostgreSQLインスタンスにインストールされ、有効になっている必要があります。インストール手順については、[Apache AGE公式ドキュメント](https://age.apache.org/docs/current/intro/installation/)を参照してください。
*   **データベース設定**: AGEが`postgresql.conf`で適切にロードおよび設定されていることを確認してください（例: `shared_preload_libraries`に`age`を追加し、PostgreSQLを再起動）。

### グラフの作成と設定
*   **グラフの存在**: `PGGraphStorage`コンポーネントは、データベース内にグラフが存在することを期待します。`postgres_impl_test.py`には`create_graph`の例がありますが、本番の`PGGraphStorage`クラス自体はグラフを自動的に作成しません。通常、`AGE_GRAPH_NAME`で指定されたグラフは既に存在するか、セットアップスクリプトによって作成されると想定されています。
    *   AGEが有効になっているデータベースに接続されたPGSQLクライアントを使用してグラフを作成できます:
        ```sql
        LOAD 'age';
        SET search_path = ag_catalog, "$user", public;
        SELECT create_graph('your_graph_name');
        ```
*   **`AGE_GRAPH_NAME`環境変数**: MiniRAGの`PGGraphStorage`で使用されるグラフの名前は、`AGE_GRAPH_NAME`環境変数を介して指定されます。MiniRAGアプリケーションが実行される環境でこの環境変数を設定する必要があります。
    ```bash
    export AGE_GRAPH_NAME="my_minirag_graph"
    ```
    または、`PGGraphStorage`が初期化される前にPythonアプリケーション内で設定します:
    ```python
    import os
    os.environ["AGE_GRAPH_NAME"] = "my_minirag_graph"
    ```
*   **検索パス**: `PGGraphStorage`の実装は、AGEの関数とオブジェクトが正しく解決されるように、操作のために検索パスを自動的に`ag_catalog, "$user", public`に設定します。

### 使用法
`PGGraphStorage`が初期化され使用されると、PostgreSQL内の指定されたグラフに対してCypherクエリを実行します。MiniRAG用に設定されたPostgreSQLの`user`が、AGEグラフに対する操作（例: クエリ、ノードやエッジの作成/更新）に必要な権限を持っていることを確認してください。

## MiniRAGでの使用法

PostgreSQLサーバーに必要な拡張機能（`pgvector`、Apache AGE）をセットアップし、接続設定の準備ができたら、MiniRAGにPostgreSQLをさまざまなストレージバックエンドとして使用するように指示できます。

MiniRAGのアーキテクチャでは、さまざまな種類のデータに対して異なるストレージメカニズムを使用できます:
*   **キーバリューストレージ (`BaseKVStorage`)**: ドキュメントの全文 (`full_docs`)、テキストチャンク (`text_chunks`)、およびLLM応答キャッシュ (`llm_response_cache`) の保存に使用されます。`PGKVStorage`によって実装されます。
*   **ベクトルストレージ (`BaseVectorStorage`)**: セマンティック検索を可能にするために、チャンク、エンティティ、およびリレーションシップのベクトル埋め込みを保存するために使用されます。`PGVectorStorage`によって実装されます。
*   **ドキュメントステータスストレージ (`DocStatusStorage`)**: ドキュメントの処理ステータスを追跡するために使用されます。`PGDocStatusStorage`によって実装されます。
*   **グラフストレージ (`BaseGraphStorage`)**: ナレッジグラフの保存とクエリに使用されます。`PGGraphStorage`によって実装されます。

### PostgreSQLを使用したMiniRAGの初期化

通常、メインの`MiniRAG`クラスまたはその基盤となるコンポーネントを初期化する際にPostgreSQLの使用を指定します。提供された設定は、PostgreSQLバックエンドのストレージクラスをインスタンス化するために使用されます。

以下は、PostgreSQLを使用するようにMiniRAGコンポーネントを初期化する方法の概念的な例です。MiniRAGアプリケーションの構成方法（例: `MiniRAG.from_config()`の使用、またはコンポーネントの手動構成）によって、正確な初期化が異なる場合があることに注意してください。

```python
import asyncio
from minirag.minirag import MiniRAG
from minirag.utils import Config
from minirag.kg.postgres_impl import (
    PostgreSQLDB,
    PGKVStorage,
    PGVectorStorage,
    PGDocStatusStorage,
    PGGraphStorage
)

# 1. PostgreSQL接続設定を定義します
postgres_config_dict = {
    "host": "localhost",
    "port": 5432,
    "user": "your_user",
    "password": "your_password",
    "database": "minirag_db",
    "workspace": "my_project_workspace"
}

# 2. (Apache AGEを使用する場合) グラフ名環境変数を設定します
import os
os.environ["AGE_GRAPH_NAME"] = "my_minirag_graph"

# 3. グローバルMiniRAG設定を作成します (YAMLファイルからもロード可能)
#    この例では、ストレージ実装クラスの設定に焦点を当てています。
#    完全なセットアップでは、LLM、埋め込みモデルなども設定します。
global_config_dict = {
    "version": "0.1.0",
    "llm_api_key": "your_llm_api_key", # または関連するLLM認証情報
    "llm_model_name": "gpt-3.5-turbo", # 例: 登録済みモデルの文字列識別子
                                         # または llm_model_func が MiniRAG コンストラクタに直接提供される場合に使用される名前。
    "embedding_model_name": "all-MiniLM-L6-v2", # 例: 登録済み埋め込みモデルの文字列識別子
                                                   # または embedding_func が直接提供される場合に使用される名前。
    "embedding_batch_num": 32,
    # ストレージ用のPostgreSQL実装を指定します
    "kv_storage_cls": "minirag.kg.postgres_impl.PGKVStorage",
    "vector_storage_cls": "minirag.kg.postgres_impl.PGVectorStorage",
    "doc_status_storage_cls": "minirag.kg.postgres_impl.PGDocStatusStorage",
    "graph_storage_cls": "minirag.kg.postgres_impl.PGGraphStorage",
    # 中央PostgreSQLデータベース設定
    "postgres_db_config": postgres_config_dict,
    # 特定のストレージクラスの引数 (dbインスタンスはMiniRAGによって渡されます)
    "vector_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    # 'db'以外の非デフォルトパラメータが必要な場合は、他の*_cls_kwargsを追加します
    # "kv_storage_cls_kwargs": {},
    # "doc_status_storage_cls_kwargs": {},
    # "graph_storage_cls_kwargs": {},
}

async def main():
    # Configオブジェクトを作成します
    config = Config(config_dict=global_config_dict)

    # MiniRAG.from_config(config) は "postgres_db_config" を使用してPostgreSQLDBの初期化を処理し、
    # dbインスタンスをストレージコンポーネントに渡します。

    # 例: KVストレージの手動初期化 (MiniRAGがこれを処理するため、説明用)
    # 注意: `namespace` と `embedding_func` はストレージクラスの説明用であり、
    # 通常は親コンポーネント (例: Indexer, Querier) によって設定されます。
    # PostgreSQLDBインスタンス `pg_db` はMiniRAGによって作成され、渡されます。

    # kv_store_fulldocs = PGKVStorage(namespace="full_docs", global_config=config, embedding_func=None, db=rag_instance.postgres_db)
    # vector_store_chunks = PGVectorStorage(namespace="chunks", global_config=config, embedding_func=my_embedding_function, db=rag_instance.postgres_db)
    # doc_status_store = PGDocStatusStorage(global_config=config, db=rag_instance.postgres_db)
    # graph_store = PGGraphStorage(namespace="my_graph_namespace", global_config=config, embedding_func=None, db=rag_instance.postgres_db)


    # 通常、MiniRAGを初期化すると、これらのストレージが設定に基づいてセットアップされます。
    # rag_instance = MiniRAG.from_config(config=config)
    # await rag_instance.init_async() # これにより、特にDB接続が初期化されます

    # 初期化後、通常どおりMiniRAGを使用してインデックス作成とクエリを実行できます。
    # 例 (概念的):
    # await rag_instance.indexer.run_pipeline_async(docs_data)
    # results = await rag_instance.querier.query_async("MiniRAGとは何ですか？")
    # print(results)

    print("MiniRAGコンポーネントがPostgreSQLバックエンドで（概念的に）初期化されました。")
    print(f"PostgreSQLサーバーが {postgres_config_dict['host']}:{postgres_config_dict['port']} で実行されていることを確認してください")
    print(f"そして、データベース '{postgres_config_dict['database']}' とユーザー '{postgres_config_dict['user']}' が正しく設定されていることを確認してください。")
    print(f"グラフ操作の場合、Apache AGEがグラフ '{os.getenv('AGE_GRAPH_NAME')}' でアクティブである必要があります。")

    # 処理が完了したらプールを閉じます (実際のアプリケーションでは重要)
    # rag_instanceを使用する場合:
    # if rag_instance.postgres_db and rag_instance.postgres_db.pool:
    #     await rag_instance.postgres_db.pool.close()

if __name__ == "__main__":
    # PGVectorStorageで必要な場合、この例のためにダミーの埋め込み関数を定義します
    # async def my_embedding_function(texts: list[str]):
    #     # 実際の埋め込みモデル呼び出しに置き換えてください
    #     print(f"Embedding {len(texts)} texts (dummy).")
    #     # 例: 適切な次元のダミーベクトルのリストを返します
    #     return [[0.1] * 768 for _ in texts]

    asyncio.run(main())
```

この例は以下を示しています:
1.  PostgreSQL接続辞書の定義。
2.  `AGE_GRAPH_NAME`環境変数の設定。
3.  PostgreSQL接続用の中央`postgres_db_config`を指定する`Config`オブジェクトの作成。MiniRAGはこれを使用して`PostgreSQLDB`インスタンスを初期化し、指定されたPostgreSQLバックエンドのストレージクラス（`PGKVStorage`、`PGVectorStorage`など）に渡します。また、これらのクラスに必要なその他の引数をそれぞれの`*_cls_kwargs`セクションで指定します。
4.  初期化を示すための`main`非同期関数。完全なMiniRAGアプリケーションでは、通常、`MiniRAG.from_config()`が提供された設定に基づいてこれらのストレージバックエンドのインスタンス化を処理します。


**モデル指定に関する注意:**

設定辞書内の`llm_model_name`および`embedding_model_name`フィールドは、通常、MiniRAGが特に`MiniRAG.from_config()`を使用する際に、事前に登録されたモデルや設定を検索するために使用する文字列識別子です。

ただし、カスタムモデル関数（`hf_model_complete`など）や、埋め込み用のラムダ関数を持つ`EmbeddingFunc`インスタンスを使用するなど、より複雑な設定の場合（例えば、直接的なPythonスクリプト、例: `main.py`で定義するような場合）:

```python
# カスタム関数とPostgreSQL用のConfigオブジェクトを使用した
# MiniRAGの直接インスタンス化の例:
#
# (postgres_connection_details と config_for_minirag が上記のように定義されていると仮定)
#
# rag = MiniRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=hf_model_complete, # 直接の関数ハンドル
#     llm_model_max_token_size=200,
#     llm_model_name=LLM_MODEL, # 参照/ロギング用の名前
#     embedding_func=EmbeddingFunc( # 直接のEmbeddingFuncインスタンス
#         embedding_dim=384,
#         max_token_size=1000,
#         func=lambda texts: hf_embed(
#             texts,
#             tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
#             embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
#         ),
#     ),
#     config=config_for_minirag # DBおよびその他の設定用のConfigオブジェクトを渡します
# )
```

このようなPostgreSQL設定との組み合わせには、いくつかのアプローチがあります:

1.  **`MiniRAG`の直接インスタンス化**: カスタム呼び出し可能オブジェクト（`llm_model_func`や`embedding_func`など）の場合、これが最も簡単なことが多いです。このアプローチを取る場合、`MiniRAG`コンストラクタに`Config`オブジェクトを渡すことでPostgreSQLデータベース設定を統合することを推奨します。この`Config`オブジェクトには`postgres_db_config`辞書が含まれている必要があります。`MiniRAG`はその後、内部的に`PostgreSQLDB`インスタンスを管理します。

    これを構造化する方法は次のとおりです:

    ```python
    # 1. PostgreSQL接続辞書を定義します
    postgres_connection_details = {
        "host": "localhost",
        "port": 5432,
        "user": "your_user",
        "password": "your_password",
        "database": "minirag_db",
        "workspace": "my_project_workspace"
    }

    # 2. PostgreSQLデータベース設定を含む、MiniRAG全体の
    #    設定用の辞書を作成します。
    #    ここには他の必要なMiniRAG設定も含めます。
    minirag_config_dict = {
        "postgres_db_config": postgres_connection_details,
        # 必要に応じて他のMiniRAGトップレベル設定を追加します。例:
        # "kv_storage_cls": "minirag.kg.postgres_impl.PGKVStorage",
        # "vector_storage_cls": "minirag.kg.postgres_impl.PGVectorStorage",
        # "doc_status_storage_cls": "minirag.kg.postgres_impl.PGDocStatusStorage",
        # "graph_storage_cls": "minirag.kg.postgres_impl.PGGraphStorage",
        # など、MiniRAGコンストラクタがこれらを使用してストレージタイプを選択する場合。
        # ただし、完全なPGバックエンドの場合、postgres_db_config が存在すればこれらはしばしばデフォルト設定されます。
    }

    # 3. Configオブジェクトを作成します
    from minirag.utils import Config
    config_for_minirag = Config(config_dict=minirag_config_dict)

    # 4. (Apache AGEを使用する場合) グラフ名環境変数を設定します
    import os
    os.environ["AGE_GRAPH_NAME"] = "my_minirag_graph" # または設定から取得します

    # 5. カスタムモデル関数とConfigオブジェクトを使用してMiniRAGを直接
    #    インスタンス化します。
    # rag = MiniRAG(
    #     working_dir=WORKING_DIR,  # 作業ディレクトリ
    #     llm_model_func=your_hf_model_complete_function, # 直接の関数ハンドル
    #     llm_model_name="your_llm_model_name_for_logging", # ロギング用のLLMモデル名
    #     embedding_func=your_embedding_func_instance, # 直接のEmbeddingFuncインスタンス
    #     config=config_for_minirag # ここにConfigオブジェクトを渡します
    # )
    #
    # # await rag.init_async() # MiniRAGを初期化します (DB初期化を含む)
    ```
    この設定では、`MiniRAG`のコンストラクタまたはその初期化プロセス（例: `init_async()`内）が、渡された`config`オブジェクト内の`postgres_db_config`を検出し、内部的に`PostgreSQLDB`インスタンスをセットアップします。これにより、`MiniRAG`のインスタンス化がクリーンに保たれ、PostgreSQLがバックエンドとして正しく設定されることが保証されます。モデル用のカスタムPythonオブジェクトは直接渡され、最大限の柔軟性が提供されます。

2.  **`MiniRAG.from_config()`の使用**: 主に`MiniRAG.from_config()`を使用したい場合、`global_config_dict`には`postgres_db_config`が含まれます。モデルについては、次のいずれかを行います:
    *   `MiniRAG.from_config()`が実際の関数/クラスに解決できる`llm_model_name`および`embedding_model_name`の文字列識別子に依存します（これには、MiniRAGへの事前の登録、またはサポートされている場合は完全修飾インポートパスの使用が必要になる場合があります）。
    *   一部のフレームワークでは、`from_config`が十分に高度であれば、Python辞書内で呼び出し可能オブジェクトを直接渡すことができますが、純粋なファイルベース（YAML/JSON）の設定ではあまり一般的ではありません。

このPostgreSQLドキュメントは、*ストレージバックエンド*の設定に焦点を当てています。上記の`global_config_dict`の例は、主にPostgreSQL固有の設定（`postgres_db_config`）がどこに適合するかを示しています。詳細なモデル設定戦略（直接インスタンス化 対 設定駆動型）については、モデルのセットアップと初期化に関するMiniRAGの一般ドキュメントを参照してください。

このようにMiniRAGを設定することで、選択したコンポーネントのすべてのデータ永続化および検索操作がPostgreSQLデータベースによって処理されます。
