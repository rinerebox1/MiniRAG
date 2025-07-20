#!/bin/bash
# build.sh - Dockerイメージをビルドするスクリプト

# エラーが発生した場合にスクリプトを終了するオプション
set -e

echo "Dockerイメージのビルドを開始します..."
# docker-compose を使用してイメージをビルド
docker compose build --no-cache

# ビルド結果のステータス表示
if [ $? -eq 0 ]; then
  echo "イメージのビルドが完了しました。"
else
  echo "イメージのビルドに失敗しました。ログを確認してください。"
  exit 1
fi