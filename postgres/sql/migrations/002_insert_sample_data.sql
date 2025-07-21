-- 002_sample_data.sql: サンプルデータ挿入 (製品データ + グラフノード・エッジ)

-- 必要な拡張をロード（pgvectorとAGEを当該セッションで利用可能にする）
LOAD 'vector';
LOAD 'age';

-- AGEの検索パスを設定
SET search_path = ag_catalog, "$user", public;

-- グラフが存在することを確認（既に存在する場合はエラーを無視）
DO $$
BEGIN
    BEGIN
        PERFORM create_graph('my_minirag_graph');
    EXCEPTION WHEN OTHERS THEN
        -- グラフが既に存在する場合は何もしない
        NULL;
    END;
END
$$;

-- 1. リレーショナルテーブル側の製品データ挿入
-- productsテーブルにサンプル商品を追加
INSERT INTO public.products (id, name, embedding) VALUES
  (1, '商品A', '[0.1, 0.2, 0.3]'),
  (2, '商品B', '[0.2, 0.1, 0.9]'),
  (3, '商品C', '[0.8, 0.9, 0.1]');

-- 2. グラフ側のユーザーノードと商品ノード、エッジを作成
-- ユーザ "Alice" のノードを作成
SELECT * FROM cypher('my_minirag_graph', $$
  CREATE (:User {name: 'Alice'})
$$) as (v agtype);

-- 商品ノードをグラフに追加（productsテーブルと対応付けるため product_id をプロパティに含める）
SELECT * FROM cypher('my_minirag_graph', $$
  CREATE (:Product {product_id: 1, name: '商品A'}),
         (:Product {product_id: 2, name: '商品B'}),
         (:Product {product_id: 3, name: '商品C'})
$$) as (v agtype);

-- Aliceが商品Aと商品Bを「LIKEした」というエッジを作成
-- p.product_id IN [1, 2] で商品A(ID=1)と商品B(ID=2)を対象
SELECT * FROM cypher('my_minirag_graph', $$
  MATCH (u:User {name: 'Alice'}), (p:Product)
  WHERE p.product_id IN [1, 2]
  CREATE (u)-[:LIKES]->(p)
$$) as (e agtype);

-- （補足）上記の MATCH で該当ノードを取得し、CREATEによりエッジを作成
-- エッジのラベルは "LIKES" 、方向は (Alice)->(Product) となる