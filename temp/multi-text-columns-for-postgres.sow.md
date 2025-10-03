# SOW: PostgreSQL複数テキストカラム対応とフィールド別検索機能

**作成日**: 2025-10-03  
**プロジェクト**: MiniRAG  
**作業対象**: PostgreSQLストレージの複数テキストフィールド対応

---

## 1. 背景と目的

### 1.1 現状の課題

現在のMiniRAGでは、PostgreSQLにテキストデータを保存する際、複数のフィールドを結合して単一の`content`カラムに保存しています：

**エンティティ（Entity）の場合:**
```python
# operate.py Line 350
"content": dp["entity_name"] + " " + dp["description"]
```

**リレーションシップ（Relationship）の場合:**
```python
# operate.py Line 373-376
"content": dp["keywords"] + " " + dp["src_id"] + " " + dp["tgt_id"] + " " + dp["description"]
```

この方式には以下の問題があります：
- 個別フィールドでの検索ができない（例: entity_nameのみで検索）
- フィールドの境界が不明確
- データの再利用性が低い
- 特定フィールドのみを更新することが困難

### 1.2 目標

複数のテキストカラムを用意し、各カラムで個別に検索・フィルタリングできるようにする：

1. **エンティティテーブル**: `entity_name`, `description` を個別カラムに分離
2. **リレーションシップテーブル**: `keywords`, `source_name`, `target_name`, `description` を個別カラムに分離
3. 各カラムに対して個別にベクトル検索やテキスト検索が可能
4. 既存の統合検索機能も維持（後方互換性）

---

## 2. 影響範囲分析

### 2.1 影響を受けるテーブル

#### **LIGHTRAG_VDB_ENTITY** テーブル
**現在:**
```sql
CREATE TABLE LIGHTRAG_VDB_ENTITY (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(255),
    content TEXT,                    -- 結合されたテキスト
    content_vector VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (workspace, id)
);
```

**変更後:**
```sql
CREATE TABLE LIGHTRAG_VDB_ENTITY (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(255),
    entity_name_vector VECTOR(1024),     -- 新規
    description TEXT,                     -- 新規（contentから分離）
    description_vector VECTOR(1024),      -- 新規
    content TEXT,                         -- 統合検索用に維持（後方互換性）
    content_vector VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);
```

#### **LIGHTRAG_VDB_RELATION** テーブル
**現在:**
```sql
CREATE TABLE LIGHTRAG_VDB_RELATION (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    source_id VARCHAR(255),
    target_id VARCHAR(255),
    content TEXT,                    -- 結合されたテキスト
    content_vector VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (workspace, id)
);
```

**変更後:**
```sql
CREATE TABLE LIGHTRAG_VDB_RELATION (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    source_id VARCHAR(255),
    target_id VARCHAR(255),
    keywords TEXT,                        -- 新規
    keywords_vector VECTOR(1024),         -- 新規
    source_name VARCHAR(255),             -- 新規
    source_name_vector VECTOR(1024),      -- 新規
    target_name VARCHAR(255),             -- 新規
    target_name_vector VECTOR(1024),      -- 新規
    description TEXT,                     -- 新規
    description_vector VECTOR(1024),      -- 新規
    content TEXT,                         -- 統合検索用に維持（後方互換性）
    content_vector VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);
```

### 2.2 影響を受けるファイル

| ファイル | 影響内容 |
|---------|---------|
| `minirag_app/minirag/kg/postgres_impl.py` | テーブル定義、SQL、upsert処理の変更 |
| `minirag_app/minirag/operate.py` | データ準備ロジックの変更 |
| `minirag_app/minirag/base.py` | QueryParam拡張（フィールド別検索パラメータ） |
| `minirag_app/minirag/minirag.py` | クエリAPIの拡張 |

---

## 3. 実装計画

### Phase 1: データベーススキーマ変更 ✅

**作業内容:**
1. マイグレーションSQLスクリプトの作成
2. テーブル定義（DDL）の更新
3. インデックスの追加（各vector列に対して）

**成果物:**
- `migration_multi_text_columns.sql`
- 更新された `TABLE_DEFINITIONS` in `postgres_impl.py`

**推定時間:** 2時間

---

### Phase 2: データ保存ロジックの変更 ✅

**作業内容:**
1. `operate.py` の `extract_entities()` 関数を修正
   - エンティティ: entity_name, description を個別フィールドで準備
   - リレーションシップ: keywords, src_id, tgt_id, description を個別フィールドで準備
   - 後方互換性のため、統合 content フィールドも生成

2. `postgres_impl.py` の `PGVectorStorage` クラスを修正
   - `_upsert_entities()`: 複数フィールド対応
   - `_upsert_relationships()`: 複数フィールド対応
   - 各フィールドに対する embedding 生成

**変更箇所:**
```python
# operate.py Line 337-382
# Before:
data_for_vdb = {
    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        "content": dp["entity_name"] + " " + dp["description"],
        "entity_name": dp["entity_name"],
        "metadata": ...
    }
}

# After:
data_for_vdb = {
    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        "entity_name": dp["entity_name"],
        "description": dp["description"],
        "content": dp["entity_name"] + " " + dp["description"],  # 後方互換性
        "metadata": ...
    }
}
```

**成果物:**
- 修正された `operate.py`
- 修正された `postgres_impl.py` の upsert メソッド

**推定時間:** 4時間

---

### Phase 3: 検索ロジックの拡張 ✅

**作業内容:**
1. `base.py` の `QueryParam` にフィールド別検索パラメータを追加
2. `postgres_impl.py` の `query()` メソッドを拡張
   - `search_fields` パラメータの追加
   - 複数フィールドの同時検索サポート
   - フィールド別の重み付けスコアリング

**新しいクエリインターフェース:**
```python
# 単一フィールド検索
results = await entity_vdb.query(
    query="Apple",
    top_k=10,
    search_fields=["entity_name"]  # entity_nameのみ検索
)

# 複数フィールド検索（重み付け）
results = await entity_vdb.query(
    query="technology company",
    top_k=10,
    search_fields=["entity_name", "description"],
    field_weights={"entity_name": 2.0, "description": 1.0}  # entity_nameを重視
)

# デフォルト（統合検索、後方互換性）
results = await entity_vdb.query(
    query="Apple technology",
    top_k=10
)
```

**成果物:**
- 拡張された `QueryParam` クラス
- 拡張された `PGVectorStorage.query()` メソッド

**推定時間:** 6時間

---

### Phase 4: API層の更新 ✅

**作業内容:**
1. `minirag.py` のクエリAPIを拡張
2. `minirag_server.py` のREST APIエンドポイント拡張
3. 検索パラメータのバリデーション

**新しいAPIエンドポイント:**
```python
POST /query/advanced
{
    "query": "Apple",
    "mode": "light",
    "search_config": {
        "entity_search_fields": ["entity_name"],
        "relationship_search_fields": ["keywords", "description"],
        "field_weights": {
            "entity_name": 2.0,
            "keywords": 1.5,
            "description": 1.0
        }
    }
}
```

**成果物:**
- 拡張されたクエリメソッド
- 新しいAPIエンドポイント
- APIドキュメント更新

**推定時間:** 4時間

---

### Phase 5: テストとドキュメント ✅

**作業内容:**
1. ユニットテストの作成・更新
2. 統合テストの実施
3. パフォーマンステスト
4. ドキュメント作成

**テストケース:**
- [ ] 単一フィールド検索が正しく動作
- [ ] 複数フィールド検索が正しく動作
- [ ] 重み付けスコアリングが正しく動作
- [ ] 後方互換性が維持されている（既存のquery()呼び出し）
- [ ] マイグレーションが正常に実行される
- [ ] 既存データへの影響がない

**成果物:**
- テストコード
- パフォーマンスレポート
- ユーザードキュメント

**推定時間:** 6時間

---

## 4. データマイグレーション戦略

### 4.1 新規インストールの場合
- 新しいスキーマで直接テーブルを作成

### 4.2 既存データがある場合
**Option A: 完全マイグレーション（推奨）**
```sql
-- Step 1: 新しいカラムを追加
ALTER TABLE LIGHTRAG_VDB_ENTITY 
    ADD COLUMN description TEXT,
    ADD COLUMN entity_name_vector VECTOR(1024),
    ADD COLUMN description_vector VECTOR(1024);

-- Step 2: 既存データを再インデックス（アプリケーション層で実行）
-- MiniRAGの再インデックス機能を使用
```

**Option B: 段階的マイグレーション**
- 新しいデータのみ新スキーマで保存
- 古いデータは必要に応じて再インデックス

### 4.3 ロールバック計画
```sql
-- カラム削除（必要に応じて）
ALTER TABLE LIGHTRAG_VDB_ENTITY 
    DROP COLUMN IF EXISTS description,
    DROP COLUMN IF EXISTS entity_name_vector,
    DROP COLUMN IF EXISTS description_vector;
```

---

## 5. パフォーマンス考慮事項

### 5.1 ストレージ影響
- **現在**: 1つのベクトル列（1024次元）
- **変更後**: 
  - エンティティ: 3つのベクトル列（entity_name_vector, description_vector, content_vector）
  - リレーションシップ: 5つのベクトル列
- **増加率**: 約3-5倍のストレージ使用量

### 5.2 インデックス戦略
```sql
-- 各ベクトル列にIVFFlatインデックスを作成
CREATE INDEX idx_entity_name_vector ON LIGHTRAG_VDB_ENTITY 
    USING ivfflat (entity_name_vector vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_description_vector ON LIGHTRAG_VDB_ENTITY 
    USING ivfflat (description_vector vector_cosine_ops) WITH (lists = 100);

-- 既存のcontentインデックスも維持
CREATE INDEX idx_content_vector ON LIGHTRAG_VDB_ENTITY 
    USING ivfflat (content_vector vector_cosine_ops) WITH (lists = 100);
```

### 5.3 クエリパフォーマンス
- **単一フィールド検索**: 従来より高速（小さいデータセット）
- **複数フィールド検索**: やや遅くなる可能性（複数クエリの統合）
- **統合検索**: 影響なし（既存のcontent列を使用）

---

## 6. リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| ストレージコスト増加 | 中 | content列をオプショナルにして、必要に応じて削除可能にする |
| マイグレーション失敗 | 高 | バックアップを必ず取得、ロールバックスクリプトを用意 |
| パフォーマンス低下 | 中 | 適切なインデックス戦略、キャッシング導入 |
| 後方互換性の破損 | 高 | 既存APIを維持、新機能はオプトイン方式 |
| 複雑性の増加 | 中 | 詳細なドキュメント作成、デフォルト動作はシンプルに |

---

## 7. 成功基準

- [ ] 複数のテキストカラムがPostgreSQLに正常に保存される
- [ ] 各カラムで個別に検索が可能
- [ ] 既存の統合検索機能が正常に動作（後方互換性）
- [ ] パフォーマンスが既存システムの80%以上を維持
- [ ] 全テストケースがパス
- [ ] マイグレーションが本番環境で正常に実行される

---

## 8. スケジュール

| フェーズ | 推定時間 | 依存関係 |
|---------|---------|---------|
| Phase 1: スキーマ変更 | 2時間 | なし |
| Phase 2: 保存ロジック | 4時間 | Phase 1完了後 |
| Phase 3: 検索ロジック | 6時間 | Phase 2完了後 |
| Phase 4: API層更新 | 4時間 | Phase 3完了後 |
| Phase 5: テスト・ドキュメント | 6時間 | Phase 4完了後 |
| **合計** | **22時間** | |

---

## 9. 承認

このSOWの内容について、以下の点を確認してください：

1. **スコープ**: 上記の作業範囲で問題ないか？
2. **優先度**: 全フェーズを実装するか、一部のみか？
3. **マイグレーション**: 既存データの扱いはどうするか？
4. **パフォーマンス**: ストレージ増加は許容範囲か？

---

**次のステップ**: 
承認後、Phase 1のスキーマ変更から着手します。

