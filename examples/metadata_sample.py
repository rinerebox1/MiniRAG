from minirag import MiniRAG, QueryParam
import os
import asyncio

# MiniRAGの初期化
# PostgreSQLへの接続情報を環境変数から取得
# ダミーのembedding_funcとllm_model_funcを設定
rag = MiniRAG(
    vector_storage="PGVectorStorage",
    kv_storage="PGKVStorage",
    graph_storage="PGGraphStorage",
    doc_status_storage="PGDocStatusStorage",
    embedding_func=lambda texts, **kwargs: [[0.1] * 1536 for _ in texts],
    llm_model_func=lambda prompt, **kwargs: "dummy response",
    addon_params={
        "postgres": {
            "host": os.environ.get("PG_HOST", "localhost"),
            "port": int(os.environ.get("PG_PORT", 5432)),
            "user": os.environ.get("PG_USER", "postgres"),
            "password": os.environ.get("PG_PASSWORD", "postgres"),
            "database": os.environ.get("PG_DATABASE", "postgres"),
        }
    },
)

# 登録するドキュメントとメタデータ
documents = [
    {"doc_id": "doc1", "content": "今日は東京でとても良い天気です。", "metadata": {"category": "weather", "city": "Tokyo"}},
    {"doc_id": "doc2", "content": "昨日の大阪は雨でした。", "metadata": {"category": "weather", "city": "Osaka"}},
    {"doc_id": "doc3", "content": "日本の首都は東京です。", "metadata": {"category": "geography", "country": "Japan"}},
]

contents_to_insert = [doc["content"] for doc in documents]
ids_to_insert = [doc["doc_id"] for doc in documents]
metadatas_to_insert = [doc["metadata"] for doc in documents]

async def main():
    # DBクライアントの初期化と設定
    # この部分は実際の環境に合わせて設定してください
    # from minirag.kg.postgres_impl import PostgreSQLDB
    # db_client = PostgreSQLDB(rag.addon_params["postgres"])
    # await db_client.initdb()
    # rag.set_storage_client(db_client)

    # メタデータを含めてデータを登録
    await rag.ainsert(contents_to_insert, ids=ids_to_insert, metadatas=metadatas_to_insert)
    print("データの登録が完了しました。")

    # メタデータでフィルタリングして検索
    # 1. 'weather' カテゴリのみを検索
    print("\\n'weather'カテゴリで検索:")
    query_param_weather = QueryParam(
        metadata_filter={"category": "weather"}
    )
    results_weather = await rag.aquery("今日の天気は？", param=query_param_weather)
    print(results_weather)

    # 2. 'geography' カテゴリかつ 'Japan' のみを検索
    print("\\n'geography'カテゴリかつ'Japan'で検索:")
    query_param_geo = QueryParam(
        metadata_filter={"category": "geography", "country": "Japan"}
    )
    results_geo = await rag.aquery("日本の首都は？", param=query_param_geo)
    print(results_geo)

    # 3. メタデータフィルタリングなしで検索
    print("\\nフィルタリングなしで検索:")
    results_all = await rag.aquery("東京について教えて")
    print(results_all)

if __name__ == "__main__":
    # このサンプルコードは、PostgreSQLデータベースが実行されている環境で動作します。
    # asyncio.run(main())
    print("サンプルコードの実行には、PostgreSQLデータベースが必要です。")
    print("`main`関数内のコメントアウトを解除し、DBに接続して実行してください。")
