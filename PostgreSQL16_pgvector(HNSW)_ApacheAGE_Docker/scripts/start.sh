#!/bin/bash
# start.sh - Dockerコンテナを起動するスクリプト

echo "PostgreSQL + AGE + pgvector コンテナを起動します..."

# コンテナをバックグラウンド(-d)で起動
docker compose up -d

# 起動したコンテナのログを少し表示して、正常起動を確認
echo "コンテナの起動ログ:"
docker compose logs -f --tail=10 db