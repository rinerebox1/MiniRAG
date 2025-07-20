#!/bin/bash
# stop.sh - Dockerコンテナを停止・削除するスクリプト

echo "PostgreSQL + AGE + pgvector コンテナを停止します..."

# コンテナの停止とネットワークの削除
docker compose down -v

echo "コンテナを停止しました。"