-- 002_add_text_field_to_existing_chunks.sql
-- 既存のチャンクデータに text_field メタデータを追加するマイグレーション

BEGIN;

-- Step 1: 既存チャンクに text_field="_all" を追加
UPDATE LIGHTRAG_DOC_CHUNKS
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{text_field}',
    '"_all"'::jsonb
)
WHERE metadata IS NULL 
   OR NOT (metadata ? 'text_field');

-- Step 2: インデックスの作成
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'lightrag_doc_chunks' 
        AND indexname = 'idx_chunks_text_field'
    ) THEN
        CREATE INDEX idx_chunks_text_field 
        ON LIGHTRAG_DOC_CHUNKS ((metadata->>'text_field'));
    END IF;
END $$;

COMMIT;
