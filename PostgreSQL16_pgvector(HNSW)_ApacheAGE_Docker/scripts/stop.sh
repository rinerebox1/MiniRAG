#!/bin/bash
# stop.sh - Dockerコンテナを停止・削除するスクリプト

echo "PostgreSQL + AGE + pgvector コンテナを停止します..."

# コンテナの停止とネットワークの削除（ボリュームは保持）
docker compose down

echo "コンテナを停止しました。"