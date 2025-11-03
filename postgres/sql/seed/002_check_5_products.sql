-- セッションごとに必要
LOAD 'age';

-- 1. 全ての Product ノードの数を確認します (合計8件のはず)
SELECT * FROM cypher('my_minirag_graph', $$
    MATCH (p:Product)
    RETURN count(p)
$$) as (product_count agtype);

-- 2. 新しく追加したユーザー 'Bob' と、彼が 'LIKES' している商品の関係を確認します (5件のはず)
SELECT * FROM cypher('my_minirag_graph', $$
    MATCH (u:User {name: 'Bob'})-[r:LIKES]->(p:Product)
    RETURN u.name, p.name, r.rating
$$) as (user_name agtype, product_name agtype, rating agtype);