#!/bin/bash
# init_db.sh - PostgreSQLコンテナ内データベースの初期セットアップスクリプト
# （pgvector拡張・AGE拡張の有効化、グラフ作成、初期スキーマ・データ投入、検索パス設定 等）

set -e

echo "データベースの初期セットアップを開始します..."

# .env から環境変数を読み込み（POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DBなど）
# ※ docker-compose 実行環境では.envが自動適用されるが、手動実行時の保険として明示的に読み込む
if [ -f /docker-entrypoint-initdb.d/.env-init ]; then
  source /docker-entrypoint-initdb.d/.env-init
fi

# 1. 拡張機能 pgvector と Apache AGE を有効化（CREATE EXTENSION）
echo "拡張機能(pgvector, age)をCREATE EXTENSIONで有効化します..."
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 <<-'SQL'
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS age;
SQL

# 2. AGEのグラフを作成
# SELECT * FROM ag_catalog.create_graph('my_minirag_graph'); を実行
# （戻り値はvoidなのでSELECT *にして実行だけ行う）
echo "Apache AGEのグラフ 'my_minirag_graph' を作成します..."
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 <<-'SQL'
    LOAD 'age';
    SET search_path = ag_catalog, "$user", public;
    SELECT * FROM ag_catalog.create_graph('my_minirag_graph');
SQL
echo "グラフ 'my_minirag_graph' が作成されました。"

# 3. マイグレーションSQLファイルの実行（昇順）
# ディレクトリ /migrations 内の *.sql を名前順に実行
echo "マイグレーションSQLを順次実行します..."
for file in /docker-entrypoint-initdb.d/migrations/*.sql; do
  echo "実行中: $file"
  # psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f "$file"
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -v INDEX_TYPE="$INDEX_TYPE" -f "$file"
done

# 4. search_path に ag_catalog を追加設定（利便性向上）
# 以降、このデータベースに接続するセッションでは、デフォルトで ag_catalog が検索パスに含まれる
echo "データベースに ag_catalog を search_path へ追加します..."
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -c "ALTER DATABASE \"$POSTGRES_DB\" SET search_path = ag_catalog, \"\$user\", public;"

echo "初期セットアップが完了しました。"