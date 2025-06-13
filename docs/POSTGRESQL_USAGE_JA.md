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

## Colabで PostgreSQL を使う方法

PostgreSQL、pgvector、Apache AGE の3つを含めた完全版のサンプルコード。

PostgreSQL、pgvector、Apache AGEの3つすべてを含む完全版のサンプルコードを提示します[1]。LangChainは使用せず、純粋なPythonとSQLで実装します。

## **完全なセットアップコード**

```python
# 1. システムの更新と必要なパッケージのインストール
!apt-get update
!apt-get install -y postgresql postgresql-contrib postgresql-server-dev-15
!apt-get install -y build-essential libreadline-dev zlib1g-dev flex bison git

# 2. PostgreSQLサーバーの起動
!service postgresql start

# 3. 必要なPythonライブラリのインストール
!pip install psycopg2-binary sqlalchemy asyncpg psycopg-pool "psycopg[binary,pool]"

print("✅ 基本システムのセットアップ完了")
```

## **pgvector拡張機能のインストール**

```python
# pgvectorのソースからインストール
def install_pgvector():
    print("pgvector拡張機能をインストール中...")
    
    # pgvectorのクローンとビルド
    !git clone https://github.com/pgvector/pgvector.git /tmp/pgvector
    
    import os
    os.chdir('/tmp/pgvector')
    !make
    !make install
    
    print("✅ pgvector拡張機能のインストール完了")

install_pgvector()
```

## **Apache AGE拡張機能のインストール**

```python
# Apache AGEのソースからインストール
def install_apache_age():
    print("Apache AGE拡張機能をインストール中...")
    
    # Apache AGEのクローンとビルド
    !git clone https://github.com/apache/age.git /tmp/age
    
    import os
    os.chdir('/tmp/age')
    !make install
    
    print("✅ Apache AGE拡張機能のインストール完了")

install_apache_age()
```

## **PostgreSQL設定の更新**

```python
# postgresql.confの設定を更新
def configure_postgresql():
    try:
        config_path = "/etc/postgresql/15/main/postgresql.conf"
        
        # shared_preload_librariesの設定
        !sudo sed -i "s/#shared_preload_libraries = ''/shared_preload_libraries = 'age'/" {config_path}
        
        # PostgreSQLサービスの再起動
        !sudo service postgresql restart
        
        print("✅ PostgreSQL設定を更新しました")
        
    except Exception as e:
        print(f"設定更新エラー: {e}")

configure_postgresql()
```

## **データベースとユーザーの設定**

```python
import psycopg2
from sqlalchemy import create_engine, text
import asyncpg

# データベース接続の設定
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'minirag_db',
    'user': 'postgres',
    'password': 'postgres'
}

# 基本的なデータベース設定
def setup_database():
    try:
        # postgresユーザーでの初期設定
        !sudo -u postgres psql -c "CREATE USER root WITH SUPERUSER"
        !sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
        !sudo -u postgres psql -c "CREATE DATABASE minirag_db;"
        
        print("✅ データベースとユーザーの設定完了")
        
    except Exception as e:
        print(f"データベース設定エラー: {e}")

setup_database()
```

## **拡張機能の有効化**

```python
# pgvectorとApache AGE拡張機能の有効化
def enable_extensions():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # AGE拡張機能をロード
        cursor.execute("LOAD 'age';")
        
        # 拡張機能を作成
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS age;")
        
        # search_pathの設定[2]
        cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
        
        # 拡張機能の確認
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'age');")
        extensions = cursor.fetchall()
        
        print("インストール済み拡張機能:")
        for ext in extensions:
            print(f"  - {ext[0]} (バージョン: {ext[1]})")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ 拡張機能の有効化完了")
        
    except Exception as e:
        print(f"拡張機能有効化エラー: {e}")

enable_extensions()
```

## **グラフデータベースの作成**

```python
# Apache AGE用のグラフ作成
def create_graph():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # search_pathの設定
        cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
        
        # グラフの作成
        cursor.execute("SELECT create_graph('minirag_graph');")
        
        # 作成されたグラフの確認
        cursor.execute("SELECT name FROM ag_graph;")
        graphs = cursor.fetchall()
        print(f"作成済みグラフ: {[graph[0] for graph in graphs]}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ グラフデータベースの作成完了")
        
    except Exception as e:
        print(f"グラフ作成エラー: {e}")

create_graph()
```

## **テーブル構造の作成**

```python
# ベクトル検索とグラフ用のテーブル作成
def create_tables():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # ドキュメント用テーブル（ベクトル検索対応）[5]
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(1536),  -- OpenAI embeddings用
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # ベクトル検索用インデックス
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
        """)
        
        # エンティティ用テーブル（グラフ連携用）
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            entity_type VARCHAR(100),
            properties JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # 関係性用テーブル
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id SERIAL PRIMARY KEY,
            source_entity_id INTEGER REFERENCES entities(id),
            target_entity_id INTEGER REFERENCES entities(id),
            relationship_type VARCHAR(100),
            weight FLOAT DEFAULT 1.0,
            properties JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ テーブル構造の作成完了")
        
    except Exception as e:
        print(f"テーブル作成エラー: {e}")

create_tables()
```

## **サンプルデータの挿入とテスト**

```python
import json
import numpy as np

# サンプルデータの挿入とテスト
def insert_sample_data():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 1. ベクトルデータの挿入テスト
        sample_embedding = np.random.rand(1536).tolist()  # ダミーベクトル
        
        cursor.execute("""
        INSERT INTO documents (content, embedding, metadata) 
        VALUES (%s, %s, %s)
        """, (
            "これはMiniRAGのテストドキュメントです。",
            json.dumps(sample_embedding),
            json.dumps({"type": "test", "category": "sample"})
        ))
        
        # 2. エンティティデータの挿入
        cursor.execute("""
        INSERT INTO entities (name, entity_type, properties) 
        VALUES (%s, %s, %s), (%s, %s, %s)
        """, (
            "MiniRAG", "システム", json.dumps({"description": "RAGシステム"}),
            "PostgreSQL", "データベース", json.dumps({"description": "リレーショナルデータベース"})
        ))
        
        # 3. 関係性データの挿入
        cursor.execute("""
        INSERT INTO relationships (source_entity_id, target_entity_id, relationship_type, weight) 
        VALUES (1, 2, 'USES', 0.9)
        """)
        
        # 4. Apache AGEでのグラフノード作成
        cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
        
        cursor.execute("""
        SELECT * FROM cypher('minirag_graph', $$
        CREATE (d:Document {
            id: 1,
            content: 'これはMiniRAGのテストドキュメントです',
            type: 'sample'
        })
        $$) AS (result agtype);
        """)
        
        cursor.execute("""
        SELECT * FROM cypher('minirag_graph', $$
        CREATE (s:System {
            id: 1,
            name: 'MiniRAG',
            type: 'RAG_System'
        })
        $$) AS (result agtype);
        """)
        
        # 5. グラフエッジの作成
        cursor.execute("""
        SELECT * FROM cypher('minirag_graph', $$
        MATCH (d:Document), (s:System)
        WHERE d.id = 1 AND s.id = 1
        CREATE (d)-[r:BELONGS_TO {confidence: 0.95}]->(s)
        $$) AS (result agtype);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ サンプルデータの挿入完了")
        
    except Exception as e:
        print(f"サンプルデータ挿入エラー: {e}")

insert_sample_data()
```

## **検索機能のテスト**

```python
# ベクトル検索とグラフクエリのテスト
def test_search_functionality():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("=== ベクトル検索テスト ===")
        # ベクトル類似性検索のテスト[5]
        test_vector = np.random.rand(1536).tolist()
        
        cursor.execute("""
        SELECT id, content, embedding  %s as distance
        FROM documents 
        ORDER BY embedding  %s 
        LIMIT 3;
        """, (json.dumps(test_vector), json.dumps(test_vector)))
        
        vector_results = cursor.fetchall()
        print(f"ベクトル検索結果: {len(vector_results)}件")
        for result in vector_results:
            print(f"  ID: {result[0]}, 距離: {result[2]:.4f}")
        
        print("\n=== グラフクエリテスト ===")
        # Apache AGEでのグラフクエリ
        cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
        
        cursor.execute("""
        SELECT * FROM cypher('minirag_graph', $$
        MATCH (d:Document)-[r:BELONGS_TO]->(s:System)
        RETURN d.content, r.confidence, s.name
        $$) AS (doc_content agtype, confidence agtype, system_name agtype);
        """)
        
        graph_results = cursor.fetchall()
        print(f"グラフクエリ結果: {len(graph_results)}件")
        for result in graph_results:
            print(f"  ドキュメント: {result[0]}, 信頼度: {result[1]}, システム: {result[2]}")
        
        cursor.close()
        conn.close()
        
        print("✅ 検索機能のテスト完了")
        
    except Exception as e:
        print(f"検索テストエラー: {e}")

test_search_functionality()
```

## **非同期接続のテスト**

```python
import asyncio

# asyncpgを使用した非同期操作のテスト
async def test_async_operations():
    try:
        # 非同期接続
        conn = await asyncpg.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        # 非同期でのデータ取得
        result = await conn.fetchval('SELECT version()')
        print(f"✅ 非同期接続成功: PostgreSQL {result.split()[1]}")
        
        # 非同期でのベクトル検索
        rows = await conn.fetch("""
        SELECT id, content, metadata 
        FROM documents 
        LIMIT 3
        """)
        
        print(f"✅ 非同期データ取得: {len(rows)}件のドキュメント")
        
        await conn.close()
        
    except Exception as e:
        print(f"非同期操作エラー: {e}")

# 非同期関数の実行
await test_async_operations()
```

## **最終確認とバックアップ**

```python
from google.colab import drive
import pandas as pd

# Google Driveマウント
drive.mount('/content/drive')

# 最終的な動作確認とバックアップ
def final_verification_and_backup():
    print("=== 最終確認 ===")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # 1. 拡張機能の確認
    cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'age');")
    extensions = cursor.fetchall()
    print("インストール済み拡張機能:")
    for ext in extensions:
        print(f"  ✅ {ext[0]} (バージョン: {ext[1]})")
    
    # 2. テーブルの確認
    cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    print(f"\n作成済みテーブル: {[table[0] for table in tables]}")
    
    # 3. グラフの確認
    cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
    cursor.execute("SELECT name FROM ag_graph;")
    graphs = cursor.fetchall()
    print(f"作成済みグラフ: {[graph[0] for graph in graphs]}")
    
    # 4. データ件数の確認
    cursor.execute("SELECT COUNT(*) FROM documents;")
    doc_count = cursor.fetchone()[0]
    print(f"ドキュメント件数: {doc_count}")
    
    cursor.close()
    conn.close()
    
    # 5. バックアップの作成
    backup_dir = '/content/drive/My Drive/minirag_backups'
    !mkdir -p {backup_dir}
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f"{backup_dir}/minirag_complete_backup_{timestamp}.sql"
    
    !pg_dump -h localhost -U postgres -d minirag_db > {backup_file}
    print(f"\n✅ バックアップ完了: {backup_file}")
    
    print("\n🎉 PostgreSQL + pgvector + Apache AGE の完全セットアップ成功！")

final_verification_and_backup()
```

この完全版のサンプルコードにより、**PostgreSQL**、**pgvector**、**Apache AGE**の3つすべてが統合されたMiniRAG対応環境がColabで構築できます[1]。ベクトル検索とグラフデータベース機能の両方を活用した高度なRAGシステムの開発が可能になります。

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

1.  <font color="red">【★推奨★】</font>**`MiniRAG`の直接インスタンス化**: カスタム呼び出し可能オブジェクト（`llm_model_func`や`embedding_func`など）の場合、これが最も簡単なことが多いです。このアプローチを取る場合、`MiniRAG`コンストラクタに`Config`オブジェクトを渡すことでPostgreSQLデータベース設定を統合することを推奨します。この`Config`オブジェクトには`postgres_db_config`辞書が含まれている必要があります。`MiniRAG`はその後、内部的に`PostgreSQLDB`インスタンスを管理します。

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
        # 必要に応じて他のMiniRAGトップレベル設定を追加します。
        # 例:

        # "kv_storage_cls": "minirag.kg.postgres_impl.PGKVStorage",
        # "vector_storage_cls": "minirag.kg.postgres_impl.PGVectorStorage",
        # "doc_status_storage_cls": "minirag.kg.postgres_impl.PGDocStatusStorage",
        # "graph_storage_cls": "minirag.kg.postgres_impl.PGGraphStorage",
    
        # 特定のストレージクラスの引数 (dbインスタンスはMiniRAGによって渡されます)
        "vector_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        # 'db'以外の非デフォルトパラメータが必要な場合は、他の*_cls_kwargsを追加します
        # "kv_storage_cls_kwargs": {},
        # "doc_status_storage_cls_kwargs": {},
        # "graph_storage_cls_kwargs": {},

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