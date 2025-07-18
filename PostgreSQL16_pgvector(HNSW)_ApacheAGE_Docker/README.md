PostgreSQL + pgvector + Apache AGE の Docker はできたので、次はMiniRAGをDockerで構築してみる。
最後に1つのDockerにまとめたい。


## MiniRAG の Docker

docker compose build --no-cache





===========================
## PostgreSQL + pgvector + Apache AGE の Docker

ルートディレクトリ: PostgreSQL16_pgvector(HNSW)_ApacheAGE_Docker

chmod +x scripts/build.sh
chmod +x scripts/start.sh
chmod +x scripts/stop.sh


修正したら
docker compose down -v
でやりなおし、ビルドし直す。



接続確認：
docker exec -it postgres16-age-pgvector psql -U postgres_user -d my_database -c "SELECT version();"

拡張機能の確認：
docker exec -it postgres16-age-pgvector psql -U postgres_user -d my_database -c "\dx"

## ハイブリッドクエリのテスト

ホスト側で
psql -h localhost -p 5432 -U <POSTGRES_USER> -d <POSTGRES_DB>
でログインする。（ただし、postgresql-client-common のインストールが必要）
なので、Docker内でやる場合は
「docker exec -it postgres16-age-pgvector psql -U postgres_user -d my_database」

次に
```
SELECT p.id, p.name, p.embedding FROM public.products AS p JOIN cypher('demo_graph', $$ MATCH (u:User {name: 'Alice'})-[:LIKES]->(prod:Product) RETURN prod.product_id $$) AS liked(product_id agtype) ON p.id = (liked.product_id)::INTEGER ORDER BY p.embedding <=> '[0.1, 0.1, 0.2]';
```
でハイブリッド検索してみる。コサイン距離なので演算子は「<=>」にした。
ここでいうハイブリッド検索は当初のイメージとは違う。以下のことをやっている。
「ユーザー 'Alice' が '好き' な商品を、特定のベクトル [0.1, 0.1, 0.2] に意味的に近い順（類似度が高い順）に並べて表示する」

## ポスグレサーバー起動中に5件のデータを追加してみる

ホスト側でターミナルを開いて

cd '/home/smorce/env2/work/MiniRAG/PostgreSQL16_pgvector(HNSW)_ApacheAGE_Docker'

chmod +x sql/seed/001_add_5_products.sql

cat sql/seed/001_add_5_products.sql | docker exec -i postgres16-age-pgvector psql -U postgres_user -d my_database

を実行する。これで
INSERT 0 5
 result 
が出れば正常にデータインサートできた。

次にインサートしたデータを確認するには、

docker exec -it postgres16-age-pgvector psql -U postgres_user -d my_database

# リレーショナルテーブルの確認
SELECT * FROM public.products ORDER BY id;

# グラフデータ（ノードとリレーションシップ）の確認
chmod +x sql/seed/002_check_5_products.sql

cat sql/seed/002_check_5_products.sql | docker exec -i postgres16-age-pgvector psql -U postgres_user -d my_database






## scripts/start.sh の結果解説

はい、提示されたログを見る限り、**データベースは正常にセットアップされ、起動していると判断できます。**

以下に、ログの主要なポイントを解説します。

*   **初期セットアップの成功**:
    *   `拡張機能(pgvector, age)をCREATE EXTENSIONで有効化します...`: `pgvector`と`Apache AGE`の拡張機能が問題なくインストールされています。
    *   `Apache AGEのグラフ 'demo_graph' を作成します...`: `demo_graph`という名前のグラフが正常に作成されています。
    *   `マイグレーションSQLを順次実行します...`: `001_init_schema.sql`（テーブルやインデックスの作成）と`002_insert_sample_data.sql`（サンプルデータの挿入）がエラーなく実行されています。
    *   `初期セットアップが完了しました。`: 初期化スクリプトが最後まで正常に完了したことを示しています。

*   **サーバーの再起動**:
    *   `waiting for server to shut down...`: **初期設定を反映させるために、PostgreSQLサーバーが一度シャットダウンしています。これはDockerの初期化プロセスにおける正常な動作です。**
    *   `database system is shut down`: サーバーが正常に停止しました。

*   **最終的なサーバー起動**:
    *   `PostgreSQL init process complete; ready for start up.`: 初期化プロセスが全て完了し、本番用のサーバーを起動する準備が整ったことを示しています。
    *   `starting PostgreSQL 16.9...`: PostgreSQLサーバーが起動を開始しました。
    *   `listening on IPv4 address "0.0.0.0", port 5432`: サーバーがポート`5432`で接続を待ち受けています。
    *   `database system is ready to accept connections`: **このメッセージが最も重要です。**データベースシステムが起動を完了し、クライアントからの接続を受け付ける準備が整ったことを明確に示しています。

**結論として、このログはエラーを示すものではなく、コンテナ内のPostgreSQLデータベースが正常に初期化され、現在稼働中であることを示しています。**


### プロジェクト概要

本プロジェクトは、PostgreSQL 16 上に Apache AGE 拡張と pgvector 拡張を統合し、リレーショナルデータベース機能・グラフデータベース機能・ベクトル類似検索機能を一体化した開発環境を提供します。グラフクエリ言語であるCypherとSQLを組み合わせたハイブリッドクエリに対応し、例えばソーシャルグラフとレコメンデーション（ベクトル類似度計算）を組み合わせた実験が可能です。この環境はDockerコンテナで手軽に構築・再現できるようになっており、必要なソースコードのビルド手順や初期設定も自動化されています。


project-root/
├── Dockerfile               # Dockerイメージ構築用の設定ファイル
├── compose.yaml             # Docker Compose 構成ファイル
├── .env                     # 環境変数定義のサンプルファイル（認証情報など）
├── init/                    # 初回のみ実行(/docker-entrypoint-initdb.dという仕組みを活用)
├── scripts/                 # 開発環境操作用スクリプトディレクトリ
│   ├── build.sh             # イメージビルド用スクリプト
│   ├── start.sh             # コンテナ起動用スクリプト
│   ├── stop.sh              # コンテナ停止用スクリプト
│   └── init_db.sh           # データベース初期化スクリプト（初回セットアップ）
├── sql/                     # データベース関連リソースディレクトリ
│   └── migrations/          # マイグレーションSQLファイル格納ディレクトリ
│       ├── 001_init_schema.sql           # スキーマ初期化SQL
│       └── 002_insert_sample_data.sql    # サンプルデータ挿入SQL
└── README.md                # プロジェクトの説明書（本ドキュメント）


### 前提条件

* **DockerおよびDocker Compose** がインストールされていること。開発環境の構築・起動・停止にはDocker Composeを利用します。
* ホストマシンに適切なリソース（メモリ、CPU）が確保されていること。特に本環境ではPostgreSQLに共有バッファ512MB等を割り当てているため、コンテナに1GB以上のメモリを利用できるようにしてください。

### セットアップ手順

1. リポジトリをクローンし、ディレクトリに移動します。
2. `.env.example` を参考に `.env` ファイルを作成します。少なくとも `POSTGRES_USER`（DBユーザ名）, `POSTGRES_PASSWORD`（パスワード）, `POSTGRES_DB`（デフォルトDB名）を設定してください。
3. ターミナルで `scripts/build.sh` を実行し、Dockerイメージをビルドします。ソースコードのダウンロードとビルドが行われるため、初回は数分程度かかる場合があります。ビルドが成功すると、「イメージのビルドが完了しました。」と表示されます。
4. 続いて、`scripts/start.sh` を実行し、コンテナを起動します。バックグラウンドでPostgreSQLサーバが起動し、初回起動時には拡張機能のインストールや初期データの投入（init\_db.sh経由）が自動的に行われます。`start.sh`実行時に表示されるログに、拡張機能の作成やグラフ作成のメッセージ（NOTICE等）が出力されるので確認してください。初期セットアップ処理が完了すると「初期セットアップが完了しました。」と表示されます。
5. コンテナが起動したら、`POSTGRES_USER` と `POSTGRES_PASSWORD` でデータベースに接続できます。例えば、ホストから `psql` コマンドラインクライアントを使用する場合:

   ```
   psql -h localhost -p 5432 -U <POSTGRES_USER> -d <POSTGRES_DB>
   ```

   として接続し、パスワード入力後にプロンプトが表示されます。コンテナ内部で直接操作する場合は、`docker-compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB` としても接続可能です。
6. 接続後、`CREATE EXTENSION age; LOAD 'age'; SET search_path = ag_catalog, "$user", public;` などのコマンドは **不要です**。すでに初期化スクリプトで拡張の作成と検索パス設定が行われているため、そのまま `cypher` 関数等を使用できます。例えば:

   ```sql
   SELECT * FROM cypher('demo_graph', $$ MATCH (u:User)-[r:LIKES]->(p:Product) RETURN u, p $$) as (u agtype, p agtype);
   ```

   といったクエリを実行して、格納されたグラフデータを取得できます。

### 使用方法

* **SQLクエリ**: 通常のPostgreSQLとして、リレーショナルテーブルに対するSQL操作（SELECT/INSERT/UPDATE/DELETE等）が可能です。例えば `SELECT * FROM products;` で製品テーブルの内容を確認できます。

* **グラフクエリ (Cypher)**: Apache AGE拡張により、`cypher('graph_name', $$ ... $$)` 構文でCypherクエリを実行できます。MATCH, CREATE, SETなどのCypher句を用いてグラフデータの操作・検索が可能です。例:

  ```sql
  SELECT * FROM cypher('demo_graph', $$
    MATCH (a:User {name: 'Alice'})-[:LIKES]->(p:Product)
    RETURN p.name
  $$) AS (product_name agtype);
  ```

  これはAliceがLIKEしている製品の名前を取得します。

* **ベクトル類似検索**: pgvector拡張により、ベクトル型に対して `<->`（L2距離）、`<#>`（内積の負値）、`<=>`（コサイン距離）等の演算子が使用できます。例えば:

  ```sql
  SELECT id, name 
  FROM products 
  ORDER BY embedding <-> '[0.0, 0.1, 0.2]' 
  LIMIT 1;
  ```

  は指定ベクトルに最も近い製品を返します。inner productやcosine距離を使う場合はそれぞれ `<#>` や `<=>` を使用できます。IVFFlatインデックスはデフォルトではL2距離用（vector\_l2\_ops）ですが、必要に応じて`vector_ip_ops`や`vector_cosine_ops`で内積・コサイン距離用のインデックスを作成できます。
→コサイン距離 に変更した。

* **ハイブリッドクエリ**: SQLとCypherを組み合わせた高度なクエリが可能です。前述の例のように、CypherクエリをSQLのFROM句に組み込んでJOINすることで、グラフの関係性に基づきリレーショナルデータをフィルタしつつ、さらにベクトル演算でスコアリングすることができます。これは本環境の大きな利点であり、グラフとベクトルによる複合的なデータ分析をシンプルなクエリで表現できます。

### トラブルシューティング

* **ビルドが失敗する**: ソースコードのダウンロードに失敗する場合、ネットワーク接続を確認してください。また、Apache AGEのリポジトリブランチ名が正しいか確認してください（PG16用v1.5.0は `PG16/v1.5.0-rc0` です）。`docker-compose build`のログでエラー内容を確認し、不足パッケージ等があればDockerfileに追加してください。
* **コンテナ起動時に拡張のエラー**: 稀に初回の`CREATE EXTENSION age`が失敗する場合があります。その際はコンテナログを確認してください。原因としてPostgreSQLのバージョン不整合（例：AGEビルド時のPostgreSQLと実行時のPostgreSQLが異なる）等が考えられます。本手順では公式Postgres16上にPG16対応のAGEをビルドしているため基本的に問題ありませんが、万一エラーが出た場合はコンテナを再作成（ボリューム削除含む）して再度試行してください。
* **クエリ実行時に関数が見つからない**: `cypher`関数が見つからない等のエラーが出た場合、拡張が正しく有効化されていない可能性があります。データベース内で `CREATE EXTENSION age;` が実行済みか確認してください（init\_db.shで実行しています）。また、`LOAD 'age';` を各セッションで行う必要がある旨メッセージが出た場合、`ALTER DATABASE ... SET search_path` の設定が効いているか確認してください（`SHOW search_path;`で`ag_catalog`が含まれていればOKです）。
* **パフォーマンス**: 開発用途ではデフォルト設定でも問題ありませんが、データ量が増えてクエリが遅く感じる場合、PostgreSQLの設定（共有バッファやワーカープロセス数など）を見直してください。また、pgvectorのIVFFlatインデックスはリスト数(`lists`)やプローブ数(`ivfflat.probes`)で精度と速度をトレードオフできます。IVFFlatインデックス作成後に

  ```sql
  SET ivfflat.probes = 10;
  ```

  のようにプローブ数を増やすとRecall（検索網羅率）が向上します（ただしクエリ実行がやや遅くなります）。本番用途ではデータ規模に応じて適切なパラメータを設定してください。
* **データ永続化**: コンテナ再起動後にデータが消えている場合、ボリューム設定が正しく機能していない可能性があります。docker-compose.ymlで`pgdata`ボリュームが設定され、コンテナの`/var/lib/postgresql/data`にマウントされていることを確認してください。正しく設定されていれば、`docker-compose down`でコンテナを消した後でもボリュームにデータは残り、次回`up`時にそのデータを読み込みます。

以上で、PostgreSQL + Apache AGE + pgvector を用いた統合開発環境のセットアップ手順と使用方法の説明は完了です。グラフデータベースとベクトル検索を組み合わせた強力なデータ分析を、このシンプルなDockerベース環境でぜひ試してみてください。必要に応じて公式ドキュメントやApache AGEコミュニティの情報を参照し、より高度なクエリや機能にも挑戦してみましょう。
