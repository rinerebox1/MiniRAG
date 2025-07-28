#!/bin/bash
# start.sh - Dockerコンテナを起動するスクリプト

# 引数の確認
CLEANUP_DB=false
if [ "$1" = "cleanup" ]; then
    CLEANUP_DB=true
    echo "DBクリーンアップモードで起動します..."
fi

# DBクリーンアップの実行
if [ "$CLEANUP_DB" = true ]; then
    echo "PostgreSQLデータボリュームをクリーンアップしています..."
    
    # 既存のコンテナを停止・削除
    docker compose down --volumes --rmi all
    
    # data/postgres ディレクトリを削除して再作成
    if [ -d "./data/postgres" ]; then
        echo "data/postgres ディレクトリを削除中..."
        rm -rf ./data/postgres
        echo "data/postgres ディレクトリを削除しました"
    fi
    
    echo "data/postgres ディレクトリをフル権限で作成中..."
    mkdir -p ./data/postgres
    sudo chown -R 999:999 ./data/postgres
    chmod 777 ./data/postgres

    # init スクリプト・マイグレーション用ディレクトリもUID 999に変更
    if [ -d "./postgres" ]; then
      sudo chown -R 999:999 ./postgres
    fi
    
    echo "data/postgres のクリーンアップが完了しました"
fi

echo "PostgreSQL + AGE + pgvector コンテナと MiniRAG コンテナを起動します..."

# 開発モードかどうかでcompose設定を選択
if [ "$DEV_MODE" = true ]; then
    echo "開発モードで起動中（ソースコードの変更がリアルタイムで反映されます）..."
    docker compose -f compose.yaml -f compose.dev.yml up -d
else
    echo "本番モードで起動中（イメージ内のソースコードを使用）..."
    docker compose up -d
fi

# 起動したコンテナのログを少し表示して、正常起動を確認
echo "コンテナの起動ログ:"
docker compose logs -f --tail=30 postgres
docker compose logs -f --tail=30 minirag_on_postgre