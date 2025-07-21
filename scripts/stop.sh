#!/bin/bash
# stop.sh - Dockerコンテナを停止・削除するスクリプト

echo "PostgreSQL + AGE + pgvector コンテナを停止します..."

# コンテナの停止とネットワークの削除
# --volumes で DB データが入ったボリュームも破棄
# --rmi all で古いイメージも削除し完全クリア
docker compose down --volumes --rmi all

echo "コンテナを停止しました。"