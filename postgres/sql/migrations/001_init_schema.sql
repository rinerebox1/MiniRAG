-- 001_init_schema.sql: スキーマ初期化（テーブル定義とインデックス作成）

-- 拡張機能のライブラリをロード（pgvector の関数・型を使用可能に）
LOAD 'vector';

-- プロダクト（商品）テーブル: ベクトル埋め込みを含む
-- id: 商品ID（整数 主キー）
-- name: 商品名（テキスト）
-- embedding: 商品特徴のベクトル（vector型, 次元数3の例）
CREATE TABLE public.products (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding VECTOR(3) NOT NULL    -- 3次元のベクトルを格納
);

-- デバッグ出力: 渡された INDEX_TYPE を表示
\echo 'INDEX_TYPE=' :INDEX_TYPE

-- ------------------------------------------------------------
-- psql の \if が「真偽値トークン」しか解釈できず、文字列比較式（HNSW = IVFFLAT 等）は評価できない。変数展開後に \if へ渡された文字列がブール値（true/false/1/0/on/off/yes/no）として解釈できないと、ログに出ているような “unrecognized value ... Boolean expected” エラーになります。
-- INDEX_TYPE が 'IVFFLAT' かどうかを SQL で判定し、psql変数 is_ivfflat に on/off を格納
-- （※ INDEX_TYPE が未定義の場合は空文字扱いで off になります）
SELECT CASE WHEN upper(:'INDEX_TYPE') = 'IVFFLAT' THEN 'on' ELSE 'off' END AS is_ivfflat;
\gset
-- ------------------------------------------------------------

\if :is_ivfflat
    \echo 'Creating IVFFlat index...'
    CREATE INDEX idx_products_embedding
        ON public.products
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
\else
    \echo 'Creating HNSW index (default)...'
    CREATE INDEX idx_products_embedding
        ON public.products
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
\endif

-- （補足）
-- vector_cosine_ops: ベクトルのコサイン類似度用の演算子クラス
-- lists=100: IVFFlatインデックスのリスト数（適宜チューニング可能）
-- m=16, ef_construction=64: HNSWインデックスのパラメータ（適宜チューニング可能）

-- ------------------------------------------------------------
-- 構造化ドキュメント格納用テーブル（複数テキストフィールドに対応）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.customer_orders (
    workspace TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT,
    summary TEXT,
    body TEXT,
    status TEXT,
    region TEXT,
    priority INTEGER,
    created_at TIMESTAMPTZ,
    PRIMARY KEY (workspace, doc_id),
    CHECK (priority IS NULL OR priority >= 0)
);

CREATE INDEX IF NOT EXISTS idx_customer_orders_workspace_created_at
    ON public.customer_orders (workspace, created_at DESC);

-- ------------------------------------------------------------
-- LIGHTRAG_DOC_CHUNKS メタデータの text_field 補完とインデックス整備
-- ------------------------------------------------------------
DO $do$
BEGIN
    IF to_regclass('public.lightrag_doc_chunks') IS NOT NULL THEN
        RAISE NOTICE 'Updating LIGHTRAG_DOC_CHUNKS.metadata -> text_field';
        EXECUTE $update$
            UPDATE LIGHTRAG_DOC_CHUNKS
            SET metadata = jsonb_set(
                COALESCE(metadata, '{}'::jsonb),
                '{text_field}',
                '"_all"'::jsonb
            )
            WHERE metadata IS NULL
               OR NOT (metadata ? 'text_field');
        $update$;

        IF NOT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'lightrag_doc_chunks'
              AND indexname = 'idx_chunks_text_field'
        ) THEN
            RAISE NOTICE 'Creating index idx_chunks_text_field on LIGHTRAG_DOC_CHUNKS';
            EXECUTE $create_index$
                CREATE INDEX idx_chunks_text_field
                    ON LIGHTRAG_DOC_CHUNKS ((metadata->>'text_field'));
            $create_index$;
        END IF;
    ELSE
        RAISE NOTICE 'Skipping LIGHTRAG_DOC_CHUNKS metadata update (table not found).';
    END IF;
END;
$do$;