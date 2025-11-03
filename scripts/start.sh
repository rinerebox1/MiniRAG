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
    
    # 現在のユーザーIDとグループIDを取得
    CURRENT_UID=$(id -u)
    CURRENT_GID=$(id -g)
    
    # ディレクトリの所有者を現在のユーザーに変更（Windowsエクスプローラーからアクセス可能にする）
    # グループはPostgreSQL用の999に設定し、PostgreSQLコンテナもアクセス可能にする
    sudo chown -R ${CURRENT_UID}:999 ./data/postgres
    # 777パーミッションで、所有者・グループ・その他すべてが読み書き可能にする
    chmod -R 777 ./data/postgres

    # init スクリプト・マイグレーション用ディレクトリは現在のユーザーのまま（読み取り専用で使用されるため）
    # 必要に応じて権限を調整
    if [ -d "./postgres" ]; then
      sudo chown -R ${CURRENT_UID}:${CURRENT_GID} ./postgres
      chmod -R 755 ./postgres
    fi
    
    echo "data/postgres のクリーンアップが完了しました"
else
    # cleanupモードでない場合も、既存のdata/postgresディレクトリの権限を確認・修正
    if [ -d "./data/postgres" ]; then
        CURRENT_UID=$(id -u)
        
        # 所有者が現在のユーザーでない場合、権限を修正
        if [ "$(stat -c '%u' ./data/postgres 2>/dev/null)" != "$CURRENT_UID" ]; then
            echo "既存のdata/postgresディレクトリの権限を修正中（Windowsエクスプローラーからアクセス可能にします）..."
            sudo chown -R ${CURRENT_UID}:999 ./data/postgres
            chmod -R 777 ./data/postgres
            echo "権限の修正が完了しました"
        fi
    fi
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