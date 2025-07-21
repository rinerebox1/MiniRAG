-- Apache AGE の準備 (セッションごとに必要)
LOAD 'age';

-- リレーショナルテーブル `products` に新しい商品を5件追加
-- 初期データとIDが重複しないように 4 から始めています
INSERT INTO public.products (id, name, embedding) VALUES
(4, 'Product D', '[0.4, 0.4, 0.5]'),
(5, 'Product E', '[0.5, 0.5, 0.6]'),
(6, 'Product F', '[0.6, 0.6, 0.7]'),
(7, 'Product G', '[0.7, 0.7, 0.8]'),
(8, 'Product H', '[0.8, 0.8, 0.9]');

-- グラフに新しい `Product` ノードを5件追加
SELECT * FROM cypher('my_minirag_graph', $$
    CREATE (:Product {product_id: 4, name: 'Product D'}),
           (:Product {product_id: 5, name: 'Product E'}),
           (:Product {product_id: 6, name: 'Product F'}),
           (:Product {product_id: 7, name: 'Product G'}),
           (:Product {product_id: 8, name: 'Product H'})
$$) as (result agtype);

-- 新しいユーザー 'Bob' を作成し、追加した商品5件に `:LIKES` 関係を追加
SELECT * FROM cypher('my_minirag_graph', $$
    CREATE (u:User {name: 'Bob'})
    WITH u
    MATCH (p:Product) WHERE p.product_id IN [4, 5, 6, 7, 8]
    CREATE (u)-[:LIKES {rating: 5}]->(p)
$$) as (result agtype);