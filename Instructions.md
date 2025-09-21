### 指示

```
async def run_tests():
    """RAGシステムのフィルタリング機能をテストするメイン関数"""
    # 0. RAGシステムのセットアップ
    try:
        rag_with_filter = await setup_rag_system()
        print("------------------------- RAGシステムが初期化されました！ -------------------------")
    except Exception as e:
        print(f"RAGシステムのセットアップに失敗しました: {e}")
        return

    # 1. テストデータの準備と登録
    # タイムスタンプのテストのため、登録を複数回に分ける
    print("\n[ステップ1] データの登録を開始します...")

    # データセット1
    docs1 = [
        {"doc_id": "doc1", "content": "今日は東京でとても良い天気です。", "metadata": {"category": "weather", "city": "Tokyo", "year": 2024}},
        {"doc_id": "doc2", "content": "昨日の大阪は雨でした。", "metadata": {"category": "weather", "city": "Osaka", "year": 2024}},
    ]
    await rag_with_filter.ainsert(
        input=[d["content"] for d in docs1],
        ids=[d["doc_id"] for d in docs1],
        metadatas=[d["metadata"] for d in docs1],
        overwrite=True
    )
    print("データセット1 (doc1, doc2) を登録しました。")
    
    time_after_docs1 = datetime.utcnow()  # → Postgres は UTC で解釈するため、必ず UTC にすること
    await asyncio.sleep(10)  # タイムスタンプを明確に区別するため10秒待機
    
    # データセット2
    docs2 = [
        {"doc_id": "doc3", "content": "日本の首都は東京です。最近は兵庫も主要な都市に入るか議論されています。", "metadata": {"category": "geography", "country": "Japan", "year": 2023}},
        {"doc_id": "doc4", "content": "大阪は日本の主要な都市の一つです。最近は秋田も主要な都市に入るか議論されています。", "metadata": {"category": "geography", "country": "Japan", "year": 2023}},
        {"doc_id": "doc5", "content": "What is the capital of Japan? It's Tokyo.", "metadata": {}},  # メタデータなし
    ]
    await rag_with_filter.ainsert(
        input=[d["content"] for d in docs2],
        ids=[d["doc_id"] for d in docs2],
        metadatas=[d["metadata"] for d in docs2],
        overwrite=True
    )
    print("データセット2 (doc3, doc4, doc5) を登録しました。")
```

- ainsertメソッドのinputを複数登録できるように改良したいです。まずはこのメソッドの挙動を理解してください。改良は待ってください。

### 制約条件

- まず不足があれば質問してから開始してください。
- 作業開始前にSOWを作成してから実行してください。
- 英語が分からないので今後の出力は日本語でお願いします。