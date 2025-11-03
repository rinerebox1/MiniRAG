#!/bin/bash
# start.sh - Dockerコンテナを起動するスクリプト

# 引数の確認
CLEANUP_DB=false
if [ "$1" = "cleanup" ]; then
    CLEANUP_DB=true
    echo "DBクリーンアップモードで起動します..."
fi

# sudo 実行時は SUDO_UID/SUDO_GID を優先し、ホストユーザーを特定
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"
POSTGRES_GID=999

if [ "$HOST_UID" = "0" ]; then
    echo "警告: ホストUIDが0です。rootで実行している可能性があります。所有権の設定に注意してください。" >&2
fi

echo "権限設定: ホストUID:GID=${HOST_UID}:${HOST_GID}, PostgreSQL GID=${POSTGRES_GID}"

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
    # ディレクトリの所有者をホストユーザーに変更し、PostgreSQLコンテナもアクセス可能にする
    sudo chown -R ${HOST_UID}:${POSTGRES_GID} ./data/postgres
    # 777パーミッションで、所有者・グループ・その他すべてが読み書き可能にする
    chmod -R 777 ./data/postgres

    # init スクリプト・マイグレーション用ディレクトリは現在のユーザーのまま（読み取り専用で使用されるため）
    # 必要に応じて権限を調整
    if [ -d "./postgres" ]; then
      sudo chown -R ${HOST_UID}:${HOST_GID} ./postgres
      chmod -R 755 ./postgres
    fi
    
    echo "data/postgres のクリーンアップが完了しました"
else
    # cleanupモードでない場合も、既存のdata/postgresディレクトリの権限を確認・修正
    if [ -d "./data/postgres" ]; then
        # 所有者が現在のユーザーでない場合、権限を修正
        if [ "$(stat -c '%u' ./data/postgres 2>/dev/null)" != "$HOST_UID" ]; then
            echo "既存のdata/postgresディレクトリの権限を修正中（Windowsエクスプローラーからアクセス可能にします）..."
            sudo chown -R ${HOST_UID}:${POSTGRES_GID} ./data/postgres
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

# PostgreSQLコンテナのhealthcheckが成功するまで待機（最大60秒）
echo "PostgreSQLコンテナの起動を待機中..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if docker compose ps postgres | grep -q "healthy"; then
        echo "PostgreSQLコンテナが正常起動しました"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done
echo ""

# PostgreSQLコンテナが起動した後、data/postgres の所有権を再設定
# （コンテナ内で作成されたファイルの所有権が postgres ユーザー (UID 999) になっているため）
if [ -d "./data/postgres" ]; then
    echo "data/postgres ディレクトリの所有権をホストユーザーに修正中（Windowsエクスプローラーからアクセス可能にします）..."
    sudo chown -R ${HOST_UID}:${POSTGRES_GID} ./data/postgres
    chmod -R 777 ./data/postgres
    echo "所有権の修正が完了しました"
fi

# 起動したコンテナのログを少し表示して、正常起動を確認
echo "コンテナの起動ログ:"
docker compose logs -f --tail=30 postgres
docker compose logs -f --tail=30 minirag_on_postgre