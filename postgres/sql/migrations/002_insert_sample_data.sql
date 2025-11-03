-- 002_sample_data.sql: サンプルデータ挿入 (製品 + グラフ + マルチフィールドチャンク)

-- 必要な拡張をロード（pgvectorとAGEを当該セッションで利用可能にする）
LOAD 'vector';
LOAD 'age';

-- AGEの検索パスを設定
SET search_path = ag_catalog, "$user", public;

-- グラフが存在することを確認（既に存在する場合は何もしない）
DO $$
BEGIN
    BEGIN
        PERFORM create_graph('my_minirag_graph');
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
END
$$;

-- 1. リレーショナルテーブル側の製品データ挿入
INSERT INTO public.products (id, name, embedding)
VALUES
  (1, '商品A', '[0.1, 0.2, 0.3]'),
  (2, '商品B', '[0.2, 0.1, 0.9]'),
  (3, '商品C', '[0.8, 0.9, 0.1]')
ON CONFLICT (id) DO NOTHING;

-- 2. グラフ側のユーザーノードと商品ノード、エッジを作成（MERGEで冪等化）
SELECT * FROM cypher('my_minirag_graph', $$
  MERGE (:User {name: 'Alice'})
$$) AS (v agtype);

SELECT * FROM cypher('my_minirag_graph', $$
  MERGE (:Product {product_id: 1, name: '商品A'})
  MERGE (:Product {product_id: 2, name: '商品B'})
  MERGE (:Product {product_id: 3, name: '商品C'})
$$) AS (v agtype);

SELECT * FROM cypher('my_minirag_graph', $$
  MATCH (u:User {name: 'Alice'})
  MATCH (p:Product)
  WHERE p.product_id IN [1, 2]
  MERGE (u)-[:LIKES]->(p)
$$) AS (e agtype);

-- 3. 構造化ドキュメントのサンプルデータ挿入（複数テキストフィールド対応）
WITH sample_docs AS (
    SELECT * FROM (VALUES
        ('sample_workspace', 'order-2026-plan',
         '2026年度 調達計画',
         '調達戦略の概要をまとめたサマリーです。',
         ARRAY['フェーズ1: 調達計画', 'フェーズ2: サプライヤ連絡'],
         'draft', 'APAC', 2,
         TIMESTAMPTZ '2025-10-01T08:30:00+00:00'
        ),
        ('sample_workspace', 'order-2025-contract',
         '北米向けサプライ契約',
         '北米市場での調達条件を整理したサマリー。',
         ARRAY['詳細1: 北米主要ベンダーの評価結果。', '詳細2: リスクと緩和策の一覧。'],
         'approved', 'APAC', 1,
         TIMESTAMPTZ '2025-10-02T09:15:00+00:00'
        )
    ) AS v(workspace, doc_id, title, summary, body_sections, status, region, priority, created_at)
)
INSERT INTO public.customer_orders (workspace, doc_id, title, summary, body, status, region, priority, created_at)
SELECT
    workspace,
    doc_id,
    title,
    summary,
    array_to_string(body_sections, E'\n'),
    status,
    region,
    priority,
    created_at
FROM sample_docs
ON CONFLICT (workspace, doc_id) DO UPDATE
    SET title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        body = EXCLUDED.body,
        status = EXCLUDED.status,
        region = EXCLUDED.region,
        priority = EXCLUDED.priority,
        created_at = EXCLUDED.created_at;

-- 4. 複数テキストフィールドチャンクの挿入（テーブルが存在する場合のみ実行）
DO $do$
BEGIN
    IF to_regclass('public.lightrag_doc_chunks') IS NULL THEN
        RAISE NOTICE 'Skipping LIGHTRAG_DOC_CHUNKS sample insert (table not found).';
        RETURN;
    END IF;

    EXECUTE $delete$
        DELETE FROM LIGHTRAG_DOC_CHUNKS
        WHERE workspace = 'sample_workspace'
          AND full_doc_id IN ('order-2026-plan', 'order-2025-contract');
    $delete$;

    EXECUTE $insert$
        INSERT INTO LIGHTRAG_DOC_CHUNKS (
            id, workspace, full_doc_id, chunk_order_index, tokens, content, content_vector, metadata
        ) VALUES
            ('chunk-order-2026-plan-title', 'sample_workspace', 'order-2026-plan', 0, 12,
                '2026年度 調達計画', NULL,
                jsonb_build_object('text_field', 'title', 'category', 'plan', 'region', 'APAC', 'year', 2026, 'status', 'draft')
            ),
            ('chunk-order-2026-plan-summary', 'sample_workspace', 'order-2026-plan', 1, 28,
                '調達戦略の概要をまとめたサマリーです。', NULL,
                jsonb_build_object('text_field', 'summary', 'category', 'plan', 'region', 'APAC', 'year', 2026, 'status', 'draft')
            ),
            ('chunk-order-2026-plan-body', 'sample_workspace', 'order-2026-plan', 2, 38,
                E'フェーズ1: 調達計画\nフェーズ2: サプライヤ連絡', NULL,
                jsonb_build_object('text_field', 'body', 'category', 'plan', 'region', 'APAC', 'year', 2026, 'status', 'draft')
            ),
            ('chunk-order-2026-plan-all', 'sample_workspace', 'order-2026-plan', 99, 64,
                E'2026年度 調達計画\n調達戦略の概要をまとめたサマリーです。\nフェーズ1: 調達計画\nフェーズ2: サプライヤ連絡', NULL,
                jsonb_build_object('text_field', '_all', 'category', 'plan', 'region', 'APAC', 'year', 2026, 'status', 'draft')
            ),
            ('chunk-order-2025-contract-title', 'sample_workspace', 'order-2025-contract', 0, 18,
                '北米向けサプライ契約', NULL,
                jsonb_build_object('text_field', 'title', 'category', 'supply', 'region', 'APAC', 'year', 2025, 'status', 'approved')
            ),
            ('chunk-order-2025-contract-summary', 'sample_workspace', 'order-2025-contract', 1, 30,
                '北米市場での調達条件を整理したサマリー。', NULL,
                jsonb_build_object('text_field', 'summary', 'category', 'supply', 'region', 'APAC', 'year', 2025, 'status', 'approved')
            ),
            ('chunk-order-2025-contract-body', 'sample_workspace', 'order-2025-contract', 2, 46,
                E'詳細1: 北米主要ベンダーの評価結果。\n詳細2: リスクと緩和策の一覧。', NULL,
                jsonb_build_object('text_field', 'body', 'category', 'supply', 'region', 'APAC', 'year', 2025, 'status', 'approved')
            ),
            ('chunk-order-2025-contract-all', 'sample_workspace', 'order-2025-contract', 99, 82,
                E'北米向けサプライ契約\n北米市場での調達条件を整理したサマリー。\n詳細1: 北米主要ベンダーの評価結果。\n詳細2: リスクと緩和策の一覧。', NULL,
                jsonb_build_object('text_field', '_all', 'category', 'supply', 'region', 'APAC', 'year', 2025, 'status', 'approved')
            )
        ON CONFLICT (workspace, id) DO UPDATE
            SET chunk_order_index = EXCLUDED.chunk_order_index,
                tokens = EXCLUDED.tokens,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata;
    $insert$;
END;
$do$;