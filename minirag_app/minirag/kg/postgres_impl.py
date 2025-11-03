import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Union, List, Dict, Set, Any, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Helper to adapt Python dict -> PostgreSQL jsonb via asyncpg
# ---------------------------------------------------------------------------
def _jsonb(obj: Any):
    """Return a value acceptable for a `$n::jsonb` placeholder.

    * If *obj* is ``None`` → return '{}' (empty JSON object string)
    * If it is already a ``str`` →そのまま返す（既に JSON シリアライズ済みとみなす）
    * それ以外は ``json.dumps`` で 1 回だけ文字列化する
    * ``datetime`` オブジェクトは ISO 8601 文字列に変換する
    """
    if obj is None:
        return "{}"
    if isinstance(obj, str):
        return obj
    
    def _default_serializer(value):
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    
    return json.dumps(obj, default=_default_serializer)


import pipmaster as pm

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import asyncpg
import sys
from tqdm.asyncio import tqdm as tqdm_async
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils import logger
from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
    BaseGraphStorage,
)
from datetime import datetime, timezone

if sys.platform.startswith("win"):
    import asyncio.windows_events

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class PostgreSQLDB:
    def __init__(self, config, **kwargs):
        self.pool = None
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.user = config.get("user", "postgres")
        self.password = config.get("password", None)
        self.database = config.get("database", "postgres")
        self.workspace = config.get("workspace", "default")
        self.max = 12
        self.increment = 1
        logger.info(f"Using the label {self.workspace} for PostgreSQL as identifier")

        if self.user is None or self.password is None or self.database is None:
            raise ValueError(
                "Missing database user, password, or database in addon_params"
            )

    async def initdb(self):
        try:
            self.pool = await asyncpg.create_pool(
                user=self.user,
                password=self.password,
                database=self.database,
                host=self.host,
                port=self.port,
                min_size=1,
                max_size=self.max,
            )

            logger.info(
                f"Connected to PostgreSQL database at {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to PostgreSQL database at {self.host}:{self.port}/{self.database}"
            )
            logger.error(f"PostgreSQL database error: {e}")
            raise

    async def check_tables(self):
        for k, v in TABLES.items():
            try:
                # テーブルが存在するかチェック
                await self.query("SELECT 1 FROM {k} LIMIT 1".format(k=k))

                # 既存テーブルに metadata 列が無い場合は追加
                if "metadata" in v["ddl"]:
                    await self.execute(f"ALTER TABLE {k} ADD COLUMN IF NOT EXISTS metadata JSONB;")

                # 既存テーブルに updated_at 列が無い場合は追加
                if "updated_at" in v["ddl"]:
                    await self.execute(f"ALTER TABLE {k} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP;")

                # 既存テーブルに created_at 列が無い場合は追加
                if "created_at" in v["ddl"]:
                    await self.execute(
                        f"ALTER TABLE {k} ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
                    )

            except Exception as e:
                logger.error(f"Failed to check table {k} in PostgreSQL database")
                logger.error(f"PostgreSQL database error: {e}")
                try:
                    await self.execute(v["ddl"])
                    logger.info(f"Created table {k} in PostgreSQL database")
                except Exception as e:
                    logger.error(f"Failed to create table {k} in PostgreSQL database")
                    logger.error(f"PostgreSQL database error: {e}")

        logger.info("Finished checking all tables in PostgreSQL database")

    async def query(
        self,
        sql: str,
        params: Union[dict, list] = None,
        multirows: bool = False,
        for_age: bool = False,
        graph_name: str = None,
    ) -> Union[dict, None, list[dict]]:
        async with self.pool.acquire() as connection:
            try:
                if for_age:
                    await PostgreSQLDB._prerequisite(connection, graph_name)
                if params:
                    if isinstance(params, dict):
                        rows = await connection.fetch(sql, *params.values())
                    elif isinstance(params, list):
                        rows = await connection.fetch(sql, *params)
                    else:
                        rows = await connection.fetch(sql, *params)
                else:
                    rows = await connection.fetch(sql)

                if multirows:
                    if rows:
                        columns = [col for col in rows[0].keys()]
                        data = [dict(zip(columns, row)) for row in rows]
                    else:
                        data = []
                else:
                    if rows:
                        columns = rows[0].keys()
                        data = dict(zip(columns, rows[0]))
                    else:
                        data = None
                return data
            except Exception as e:
                logger.error(f"PostgreSQL database error: {e}")
                print(sql)
                print(params)
                raise

    async def execute(
        self,
        sql: str,
        data: Union[list, dict] = None,
        for_age: bool = False,
        graph_name: str = None,
        upsert: bool = False,
    ):
        try:
            async with self.pool.acquire() as connection:
                if for_age:
                    await PostgreSQLDB._prerequisite(connection, graph_name)

                if data is None:
                    await connection.execute(sql)
                else:
                    await connection.execute(sql, *data.values())
        except (
            asyncpg.exceptions.UniqueViolationError,
            asyncpg.exceptions.DuplicateTableError,
        ) as e:
            if upsert:
                print("Key value duplicate, but upsert succeeded.")
            else:
                logger.error(f"Upsert error: {e}")
        except Exception as e:
            logger.error(f"PostgreSQL database error: {e.__class__} - {e}")
            print(sql)
            print(data)
            raise

    @staticmethod
    async def _prerequisite(conn: asyncpg.Connection, graph_name: str):
        try:
            await conn.execute('SET search_path = ag_catalog, "$user", public')
            await conn.execute(f"""select create_graph('{graph_name}')""")
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            print("★デバッグ(postgres_impl.py): create_graph already exists")
            pass


@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = None

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> Union[dict, None]:
        """Get doc_full data by id."""
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "id": id}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
        else:
            res = await self.db.query(sql, params)
        if res:
            return res
        else:
            return None

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        """Specifically for llm_response_cache."""
        sql = SQL_TEMPLATES["get_by_mode_id_" + self.namespace]
        params = {"workspace": self.db.workspace, mode: mode, "id": id}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            return res
        else:
            return None

    # Query by id
    async def get_by_ids(self, ids: List[str], fields=None) -> Union[List[dict], None]:
        """Get doc_chunks data by id"""
        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.db.workspace}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            modes = set()
            dict_res: dict[str, dict] = {}
            for row in array_res:
                modes.add(row["mode"])
            for mode in modes:
                if mode not in dict_res:
                    dict_res[mode] = {}
            for row in array_res:
                dict_res[row["mode"]][row["id"]] = row
            res = [{k: v} for k, v in dict_res.items()]
        else:
            res = await self.db.query(sql, params, multirows=True)
        if res:
            return res
        else:
            return None

    async def all_keys(self) -> list[dict]:
        if "llm_response_cache" == self.namespace:
            sql = "select workspace,mode,id from lightrag_llm_cache"
            res = await self.db.query(sql, multirows=True)
            return res
        else:
            logger.error(
                f"all_keys is only implemented for llm_response_cache, not for {self.namespace}"
            )

    async def filter_keys(self, keys: List[str]) -> Set[str]:
        """Filter out duplicated content"""
        sql = SQL_TEMPLATES["filter_keys"].format(
            table_name=NAMESPACE_TABLE_MAP[self.namespace],
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.db.workspace}
        try:
            res = await self.db.query(sql, params, multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            data = set([s for s in keys if s not in exist_keys])
            return data
        except Exception as e:
            logger.error(f"PostgreSQL database error: {e}")
            print(sql)
            print(params)

    ################ INSERT METHODS ################
    async def upsert(self, data: Dict[str, dict]):
        if self.namespace == "text_chunks":
            # text_chunks are upserted through PGVectorStorage
            pass
        elif self.namespace == "full_docs":
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.db.workspace,
                    "metadata": _jsonb(v.get("metadata", {})),
                }
                await self.db.execute(upsert_sql, _data)
        elif self.namespace == "llm_response_cache":
            for mode, items in data.items():
                for k, v in items.items():
                    upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                    _data = {
                        "workspace": self.db.workspace,
                        "id": k,
                        "original_prompt": v["original_prompt"],
                        "return_value": v["return"],
                        "mode": mode,
                        "metadata": _jsonb(v.get("metadata", {})),
                    }

                    await self.db.execute(upsert_sql, _data)

    async def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """指定されたドキュメントIDに関連するレコードを削除"""
        if not doc_ids:
            return
            
        doc_ids_str = ",".join([f"'{doc_id}'" for doc_id in doc_ids])
        
        if self.namespace == "full_docs":
            sql = f"DELETE FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id IN ({doc_ids_str})"
        elif self.namespace == "text_chunks":
            sql = f"DELETE FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND full_doc_id IN ({doc_ids_str})"
        else:
            logger.warning(f"delete_by_doc_ids not implemented for namespace: {self.namespace}")
            return
            
        await self.db.execute(sql, {"workspace": self.db.workspace})
        print(f"🗑️  Deleted {self.namespace} records for doc_ids: {doc_ids}")
        logger.info(f"Deleted {self.namespace} records for doc_ids: {doc_ids}")

    async def index_done_callback(self):
        if self.namespace in ["full_docs", "text_chunks"]:
            logger.info("full doc and chunk data had been saved into postgresql db!")

    async def get_chunk_ids_by_doc_ids(self, doc_ids: list[str]) -> list[str]:
        """PostgreSQLから指定されたdoc_idに紐づくチャンクIDを取得"""
        if not doc_ids:
            return []

        ids_str = ",".join([f"'{doc_id}'" for doc_id in doc_ids])
        sql = (
            f"SELECT id FROM LIGHTRAG_DOC_CHUNKS "
            f"WHERE workspace=$1 AND full_doc_id IN ({ids_str})"
        )

        try:
            rows = await self.db.query(sql, [self.db.workspace], multirows=True)
            if rows:
                return [row["id"] for row in rows]
            return []
        except Exception as e:
            logger.error(f"Failed to get chunk IDs by doc IDs: {e}")
            return []


@dataclass
class PGVectorStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.2"))
    db: PostgreSQLDB = None

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # Use global config value if specified, otherwise use default
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def _upsert_chunks(self, item: dict):
        try:
            upsert_sql = SQL_TEMPLATES["upsert_chunk"]
            # メタデータをJSON文字列として渡すが、データベース側でJSONBとして扱う
            metadata = _jsonb(item.get("metadata"))
            data = {
                "workspace": self.db.workspace,
                "id": item["__id__"],
                "tokens": item["tokens"],
                "chunk_order_index": item["chunk_order_index"],
                "full_doc_id": item["full_doc_id"],
                "content": item["content"],
                "content_vector": json.dumps(item["__vector__"].tolist()),
                "metadata": metadata,
            }
            print(f"💾 Upserting chunk '{item['__id__'][:16]}...' with metadata: {item.get('metadata', {})}")
        except Exception as e:
            logger.error(f"Error to prepare upsert sql: {e}")
            print(item)
            raise e
        return upsert_sql, data

    def _upsert_entities(self, item: dict):
        upsert_sql = SQL_TEMPLATES["upsert_entity"]
        data = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "metadata": _jsonb(item.get("metadata")),
        }
        return upsert_sql, data

    def _upsert_relationships(self, item: dict):
        upsert_sql = SQL_TEMPLATES["upsert_relationship"]
        data = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "metadata": _jsonb(item.get("metadata")),
        }
        return upsert_sql, data

    async def upsert(self, data: Dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        for item in list_data:
            if self.namespace == "chunks":
                upsert_sql, data = self._upsert_chunks(item)
            elif self.namespace == "entities_name":
                # 修正: "entities_name"は"entities"と同じデータ構造として扱う
                upsert_sql, data = self._upsert_entities(item)
            elif self.namespace == "entities":
                upsert_sql, data = self._upsert_entities(item)
            elif self.namespace == "relationships":
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    async def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """指定されたドキュメントIDに関連するベクターレコードを削除"""
        if not doc_ids:
            return
            
        doc_ids_str = ",".join([f"'{doc_id}'" for doc_id in doc_ids])
        
        table_name = NAMESPACE_TABLE_MAP.get(self.namespace)
        if not table_name:
            logger.warning(f"delete_by_doc_ids not implemented for namespace: {self.namespace}")
            return

        # チャンクとフルドキュメントは `full_doc_id` で削除
        if self.namespace in ["chunks", "full_docs"]:
            id_column = "full_doc_id"
        else:
            # その他のVDBは直接 `id` で削除するが、通常このケースは限定的
            id_column = "id"

        sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND {id_column} IN ({doc_ids_str})"
            
        await self.db.execute(sql, [self.db.workspace])
        print(f"🗑️  Deleted {self.namespace} vector records for doc_ids: {doc_ids}")
        logger.info(f"Deleted {self.namespace} vector records for doc_ids: {doc_ids}")

    async def index_done_callback(self):
        logger.info("vector data had been saved into postgresql db!")

    #################### query method ###############
    async def query(
        self,
        query: str,
        top_k=5,
        metadata_filter: dict = None,
        start_time: str = None,
        end_time: str = None,
        debug: bool = True,
    ) -> Union[dict, list[dict]]:
        """向量数据库を検索"""
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]
        embedding_string = ",".join(map(str, embedding))

        base_sql = SQL_TEMPLATES[self.namespace]
        
        # WHERE句を動的に構築
        where_clauses = ["workspace=$1", "distance>$2"]
        params = [self.db.workspace, self.cosine_better_than_threshold]
        if debug:
            print(f"🎯 Using distance threshold: {self.cosine_better_than_threshold}")
        
        param_idx = 3 # パラメータインデックスは$3から開始

        if metadata_filter:
            for key, value in metadata_filter.items():
                # メタデータの型に応じて適切なアクセス方法を選択
                # JSONB object型の場合は直接アクセス、string型の場合は変換が必要
                
                # 🆕 リスト値の場合は IN 句を生成（複数フィールド検索用）
                if isinstance(value, list):
                    placeholders = ",".join([f"${param_idx + i}" for i in range(len(value))])
                    where_clauses.append(f"""metadata IS NOT NULL AND (
                        (jsonb_typeof(metadata) = 'object' AND metadata->>'{key}' IN ({placeholders})) OR
                        (jsonb_typeof(metadata) = 'string' AND (metadata::text)::jsonb->>'{key}' IN ({placeholders}))
                    )""")
                    params.extend([str(v) for v in value])
                    param_idx += len(value)
                    if debug:
                        print(f"🔧 Flexible metadata filter (IN): {key} IN {value} (handles both object and string types)")
                elif isinstance(value, (int, float)):
                    # 両方のケースに対応（object型とstring型）
                    where_clauses.append(f"""metadata IS NOT NULL AND (
                        (jsonb_typeof(metadata) = 'object' AND (metadata->>'{key}')::numeric = ${param_idx}) OR
                        (jsonb_typeof(metadata) = 'string' AND ((metadata::text)::jsonb->>'{key}')::numeric = ${param_idx})
                    )""")
                    params.append(value)
                    param_idx += 1
                    if debug:
                        print(f"🔧 Flexible metadata filter: {key} = {str(value)} (handles both object and string types)")
                else:
                    where_clauses.append(f"""metadata IS NOT NULL AND (
                        (jsonb_typeof(metadata) = 'object' AND metadata->>'{key}' = ${param_idx}) OR
                        (jsonb_typeof(metadata) = 'string' AND (metadata::text)::jsonb->>'{key}' = ${param_idx})
                    )""")
                    params.append(str(value))
                    param_idx += 1
                    if debug:
                        print(f"🔧 Flexible metadata filter: {key} = {str(value)} (handles both object and string types)")
        
        if start_time:
            # datetimeオブジェクトまたは文字列を処理
            if isinstance(start_time, datetime):
                # タイムゾーン付きの場合はUTCに正規化
                if start_time.tzinfo is not None:
                    start_time = start_time.astimezone(timezone.utc)
                # naiveなdatetimeとして扱う（PostgreSQLのTIMESTAMP WITHOUT TIME ZONEに対応）
                start_time = start_time.replace(tzinfo=None)
            elif isinstance(start_time, str):
                # ISO形式の文字列をdatetimeオブジェクトにパース
                try:
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    # タイムゾーン付きの場合はUTCに正規化
                    if start_time.tzinfo is not None:
                        start_time = start_time.astimezone(timezone.utc)
                    # naiveなdatetimeに変換
                    start_time = start_time.replace(tzinfo=None)
                except (ValueError, AttributeError):
                    # パースできない場合はエラーを発生させる
                    raise ValueError(f"Invalid datetime string format: {start_time}")
            else:
                # その他の型はエラー
                raise TypeError(f"start_time must be datetime or ISO format string, got {type(start_time)}")
            
            where_clauses.append(f"updated_at >= ${param_idx}::timestamp")
            params.append(start_time)
            param_idx += 1

        if end_time:
            # datetimeオブジェクトまたは文字列を処理
            if isinstance(end_time, datetime):
                # タイムゾーン付きの場合はUTCに正規化
                if end_time.tzinfo is not None:
                    end_time = end_time.astimezone(timezone.utc)
                # naiveなdatetimeとして扱う（PostgreSQLのTIMESTAMP WITHOUT TIME ZONEに対応）
                end_time = end_time.replace(tzinfo=None)
            elif isinstance(end_time, str):
                # ISO形式の文字列をdatetimeオブジェクトにパース
                try:
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    # タイムゾーン付きの場合はUTCに正規化
                    if end_time.tzinfo is not None:
                        end_time = end_time.astimezone(timezone.utc)
                    # naiveなdatetimeに変換
                    end_time = end_time.replace(tzinfo=None)
                except (ValueError, AttributeError):
                    # パースできない場合はエラーを発生させる
                    raise ValueError(f"Invalid datetime string format: {end_time}")
            else:
                # その他の型はエラー
                raise TypeError(f"end_time must be datetime or ISO format string, got {type(end_time)}")
            
            where_clauses.append(f"updated_at <= ${param_idx}::timestamp")
            params.append(end_time)
            param_idx += 1

        # SQLクエリを組み立て
        sql = base_sql.format(
            embedding_string=embedding_string,
            where_clause=" AND ".join(where_clauses)
        )
        
        # LIMIT句を追加
        sql += f" ORDER BY distance DESC LIMIT ${param_idx}"
        params.append(top_k)

        # デバッグ情報を追加
        if metadata_filter:
            if debug:
                print(f"🔍 Metadata filter applied: {metadata_filter}")
                print(f"🔍 Generated SQL: {sql}")
                print(f"🔍 Parameters: {params}")
        else:
            if debug and self.namespace == "chunks":
                print("🔍 No metadata filter applied")

        # クエリ実行
        try:
            results = await self.db.query(sql, params, multirows=True)
            if debug:
                print(f"📊 Query returned {len(results) if results else 0} results")
            
            # 結果を表示（最初の2件）
            if debug and results:
                for i, result in enumerate(results[:2]):
                    print(f"✅ Result {i+1}: id={result.get('id', '')[:16]}..., distance={result.get('distance', 'N/A')}")
            
            # データベースの状況を確認（名前空間ごとにテーブルを判定）
            if debug:
                table_name = NAMESPACE_TABLE_MAP.get(self.namespace, "LIGHTRAG_DOC_CHUNKS")
                debug_sql = f"SELECT COUNT(*) as total FROM {table_name} WHERE workspace=$1"
                count_result = await self.db.query(debug_sql, [self.db.workspace])
                total_records = count_result["total"] if count_result else 0

                print(f"🔎 Total records in {table_name}: {total_records}")

                # メタデータ列が存在する前提で件数を確認
                meta_sql = (
                    f"SELECT COUNT(*) as with_meta FROM {table_name} "
                    "WHERE workspace=$1 AND metadata IS NOT NULL AND metadata != '{}'::jsonb"
                )
                meta_result = await self.db.query(meta_sql, [self.db.workspace])
                with_meta = meta_result["with_meta"] if meta_result else 0
                if self.namespace == "chunks":
                    print(f"🔎 Records with metadata: {with_meta}")

                # メタデータを持つレコードの例（上位3件）と distance を確認
                if with_meta > 0:
                    distance_sql = (
                        f"SELECT id, metadata, "
                        "CASE "
                        "  WHEN jsonb_typeof(metadata) = 'object' THEN metadata->>'category' "
                        "  WHEN jsonb_typeof(metadata) = 'string' THEN (metadata::text)::jsonb->>'category' "
                        "  ELSE NULL "
                        "END as category, "
                        f"1 - (content_vector <=> '[{embedding_string}]'::vector) as distance "
                        f"FROM {table_name} "
                        "WHERE workspace=$1 AND metadata IS NOT NULL "
                        "ORDER BY distance DESC LIMIT 3"
                    )
                    distance_results = await self.db.query(distance_sql, [self.db.workspace], multirows=True)
                    if debug:
                        print("🔎 Raw metadata and distance values:")
                        for dr in distance_results:
                            metadata_raw = dr.get('metadata')
                            # メタデータが文字列の場合はJSONとしてパースして表示
                            # これにより、実際にメタデータが登録されている場合は正しく表示される
                            if isinstance(metadata_raw, str):
                                try:
                                    metadata_parsed = json.loads(metadata_raw)
                                    if isinstance(metadata_parsed, dict) and metadata_parsed:
                                        print(f"   - ID: {dr.get('id', '')[:16]}...")
                                        print(f"     Raw metadata: {metadata_parsed}")
                                        print(f"     Extracted category: {dr.get('category')}")
                                        print(f"     Distance: {dr.get('distance')}")
                                    else:
                                        # 空の辞書の場合は表示をスキップ（元々メタデータが登録されていない場合）
                                        print(f"   - ID: {dr.get('id', '')[:16]}...")
                                        print(f"     Raw metadata: {{}} (no metadata registered)")
                                        print(f"     Distance: {dr.get('distance')}")
                                except json.JSONDecodeError:
                                    print(f"   - ID: {dr.get('id', '')[:16]}...")
                                    print(f"     Raw metadata: {metadata_raw} (parse error)")
                                    print(f"     Distance: {dr.get('distance')}")
                            elif isinstance(metadata_raw, dict):
                                if metadata_raw:
                                    print(f"   - ID: {dr.get('id', '')[:16]}...")
                                    print(f"     Raw metadata: {metadata_raw}")
                                    print(f"     Extracted category: {dr.get('category')}")
                                    print(f"     Distance: {dr.get('distance')}")
                                else:
                                    # 空の辞書の場合は表示をスキップ（元々メタデータが登録されていない場合）
                                    print(f"   - ID: {dr.get('id', '')[:16]}...")
                                    print(f"     Raw metadata: {{}} (no metadata registered)")
                                    print(f"     Distance: {dr.get('distance')}")
                            else:
                                print(f"   - ID: {dr.get('id', '')[:16]}...")
                                print(f"     Raw metadata: {metadata_raw}")
                                print(f"     Metadata type: {type(metadata_raw)}")
                                print(f"     Distance: {dr.get('distance')}")
            
            return results
        except Exception as e:
            logger.error(f"Error executing vector query with metadata filter: {e}")
            logger.error(f"SQL: {sql}")
            logger.error(f"Parameters: {params}")
            raise

    async def delete_by_ids(self, record_ids: list[str]) -> None:
        """Delete vectors by their primary IDs (workspace との複合 PK)。

        Args:
            record_ids: 物理テーブルの id カラムに対応する ID 群
        """
        if not record_ids:
            return

        ids_str = ",".join([f"'{rid}'" for rid in record_ids])

        if self.namespace == "chunks":
            table = "LIGHTRAG_DOC_CHUNKS"
        elif self.namespace in ["entities", "entities_name"]:
            table = "LIGHTRAG_VDB_ENTITY"
        elif self.namespace == "relationships":
            table = "LIGHTRAG_VDB_RELATION"
        else:
            logger.warning(f"delete_by_ids not implemented for namespace: {self.namespace}")
            return

        sql = f"DELETE FROM {table} WHERE workspace=$1 AND id IN ({ids_str})"
        await self.db.execute(sql, [self.db.workspace])
        print(f"🗑️  Deleted {self.namespace} vector records by ids: {record_ids[:3]}... (total {len(record_ids)})")
        logger.info(f"Deleted {self.namespace} vector records by ids: {len(record_ids)}")

    async def delete_entity(self, entity_name: str):
        """Delete entity vectors (both 'entities' と 'entities_name')."""
        if self.namespace not in ["entities", "entities_name"]:
            logger.warning("delete_entity called on non-entity namespace %s", self.namespace)
            return

        table = "LIGHTRAG_VDB_ENTITY"
        sql = f"DELETE FROM {table} WHERE workspace=$1 AND entity_name=$2"
        await self.db.execute(sql, [self.db.workspace, entity_name])
        print(f"🗑️  Deleted entity vectors for {entity_name}")
        logger.info("Deleted entity vectors for %s", entity_name)

    async def delete_relation(self, entity_name: str):
        """Delete relationship vectors whose src_id or tgt_id equals the entity."""
        if self.namespace != "relationships":
            logger.warning("delete_relation called on non-relationship namespace %s", self.namespace)
            return

        table = "LIGHTRAG_VDB_RELATION"
        sql = (
            f"DELETE FROM {table} WHERE workspace=$1 AND (source_id=$2 OR target_id=$2)"
        )
        await self.db.execute(sql, [self.db.workspace, entity_name])
        print(f"🗑️  Deleted relationship vectors involving {entity_name}")
        logger.info("Deleted relationship vectors involving %s", entity_name)


@dataclass
class PGDocStatusStorage(DocStatusStorage):
    """PostgreSQL implementation of document status storage"""

    db: PostgreSQLDB = None

    def __post_init__(self):
        pass

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return keys that don't exist in storage"""
        keys = ",".join([f"'{_id}'" for _id in data])
        sql = (
            f"SELECT id FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND id IN ({keys})"
        )
        result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
        # The result is like [{'id': 'id1'}, {'id': 'id2'}, ...].
        if result is None:
            return set(data)
        else:
            existed = set([element["id"] for element in result])
            return set(data) - existed

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of documents in each status"""
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  where workspace=$1 GROUP BY STATUS
                 """
        result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
        # Result is like [{'status': 'PENDING', 'count': 1}, {'status': 'PROCESSING', 'count': 2}, ...]
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        """Get all documents by status"""
        # 修正点1: SQLのプレースホルダーを $1 と $2 に修正
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and status=$2"
        params = {"workspace": self.db.workspace, "status": status}
        db_result = await self.db.query(sql, params, True)
        # Result is like [{'id': 'id1', 'status': 'PENDING', 'updated_at': '2023-07-01 00:00:00'}, {'id': 'id2', 'status': 'PENDING', 'updated_at': '2023-07-01 00:00:00'}, ...]
        # Converting to be a dict

        processed_docs = {}
        if db_result:
            for element in db_result:
                content = element.get("content")
                if not content:
                    content = element.get("content_summary", "")

                processed_docs[element["id"]] = DocProcessingStatus(
                    content=content,
                    content_summary=element["content_summary"],
                    content_length=element["content_length"],
                    status=element["status"],
                    created_at=element["created_at"],
                    updated_at=element["updated_at"],
                    chunks_count=element["chunks_count"],
                    metadata=element.get("metadata", {}),
                )
        return processed_docs

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        return await self.get_docs_by_status(DocStatus.FAILED)

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        return await self.get_docs_by_status(DocStatus.PENDING)

    async def index_done_callback(self):
        """Save data after indexing, but for PostgreSQL, we already saved them during the upsert stage, so no action to take here"""
        logger.info("Doc status had been saved into postgresql db!")

    async def upsert(self, data: dict[str, dict]):
        """Update or insert document status

        Args:
            data: Dictionary of document IDs and their status data
        """
        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status, metadata)
                 values($1,$2,$3,$4,$5,$6, $7::jsonb)
                  on conflict(id,workspace) do update set
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  metadata = EXCLUDED.metadata,
                  updated_at = CURRENT_TIMESTAMP"""
        for k, v in data.items():
            # chunks_count is optional
            await self.db.execute(
                sql,
                {
                    "workspace": self.db.workspace,
                    "id": k,
                    "content_summary": v["content_summary"],
                    "content_length": v["content_length"],
                    "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                    "status": v["status"],
                    "metadata": _jsonb(v.get("metadata", {})),
                },
            )
        return data


class PGGraphQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@dataclass
class PGGraphStorage(BaseGraphStorage):
    db: PostgreSQLDB = None

    @staticmethod
    def load_nx_graph(file_name):
        print("本番ではAGEを使ったグラフのプリロードは行わない")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.graph_name = os.environ["AGE_GRAPH_NAME"]
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    def __post_init__(self):
        # AGEグラフデータベースではプリロードは行わない
        # グラフデータはAGEデータベースに直接保存・読み込みされる
        logger.info(
            f"PGGraphStorage initialized for namespace '{self.namespace}' with AGE graph '{self.graph_name}'"
        )

    async def index_done_callback(self):
        print("KG successfully indexed.")

    @staticmethod
    def _record_to_dict(record: asyncpg.Record) -> Dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record.keys():
            v = record[k]
            # agtype comes back '{key: value}::type' which must be parsed
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
                if dtype == "vertex":
                    vertex = json.loads(v)
                    vertices[vertex["id"]] = vertex.get("properties")

        # iterate returned fields and parse appropriately
        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
            else:
                dtype = ""

            if dtype == "vertex":
                vertex = json.loads(v)
                field = vertex.get("properties")
                if not field:
                    field = {}
                field["label"] = PGGraphStorage._decode_graph_label(field["node_id"])
                d[k] = field
            # convert edge from id-label->id by replacing id with node information
            # we only do this if the vertex was also returned in the query
            # this is an attempt to be consistent with neo4j implementation
            elif dtype == "edge":
                edge = json.loads(v)
                d[k] = (
                    vertices.get(edge["start_id"], {}),
                    edge[
                        "label"
                    ],  # we don't use decode_graph_label(), since edge label is always "DIRECTED"
                    vertices.get(edge["end_id"], {}),
                )
            else:
                d[k] = json.loads(v) if isinstance(v, str) else v

        return d

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], _id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            _id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def _encode_graph_label(label: str) -> str:
        """
        Since AGE supports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            label (str): the original label

        Returns:
            str: the encoded label
        """
        return "x" + label.encode().hex()

    @staticmethod
    def _decode_graph_label(encoded_label: str) -> str:
        """
        Since AGE supports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            encoded_label (str): the encoded label

        Returns:
            str: the decoded label
        """
        return bytes.fromhex(encoded_label.removeprefix("x")).decode()

    @staticmethod
    def _get_col_name(field: str, idx: int) -> str:
        """
        Convert a cypher return field to a pgsql select field
        If possible keep the cypher column name, but create a generic name if necessary

        Args:
            field (str): a return field from a cypher query to be formatted for pgsql
            idx (int): the position of the field in the return statement

        Returns:
            str: the field to be used in the pgsql select statement
        """
        # remove white space
        field = field.strip()
        # if an alias is provided for the field, use it
        if " as " in field:
            return field.split(" as ")[-1].strip()
        # if the return value is an unnamed primitive, give it a generic name
        if field.isnumeric() or field in ("true", "false", "null"):
            return f"column_{idx}"
        # otherwise return the value stripping out some common special chars
        return field.replace("(", "_").replace(")", "")

    async def _query(
        self, query: str, readonly: bool = True, upsert: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        # convert cypher query to pgsql/age query
        wrapped_query = query

        # execute the query, rolling back on an error
        try:
            if readonly:
                data = await self.db.query(
                    wrapped_query,
                    multirows=True,
                    for_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await self.db.execute(
                    wrapped_query,
                    for_age=True,
                    graph_name=self.graph_name,
                    upsert=upsert,
                )
        except Exception as e:
            raise PGGraphQueryException(
                {
                    "message": f"Error executing graph query: {query}",
                    "wrapped": wrapped_query,
                    "detail": str(e),
                }
            ) from e

        if data is None:
            result = []
        # decode records
        else:
            result = [PGGraphStorage._record_to_dict(d) for d in data]

        return result

    # 修正: エンティティの種類を取得するメソッド（AGE対応）にして追加
    async def get_types(self):
        types = set()
        types_with_case = set()

        # AGEグラフデータベースからエンティティタイプを取得
        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity)
                     RETURN DISTINCT properties(n).entity_type AS entity_type
                   $$) AS (entity_type agtype)""" % (self.graph_name,)
        
        try:
            records = await self._query(query)
            for record in records:
                if record and record.get("entity_type"):
                    entity_type = record["entity_type"]
                    if entity_type:
                        types.add(entity_type.lower())
                        types_with_case.add(entity_type)
        except Exception as e:
            logger.error(f"Error getting entity types: {e}")
            
        return list(types), list(types_with_case)

    # 修正: 指定したタイプのノードを取得するメソッド（AGE対応）にして追加
    async def get_node_from_types(self, type_list) -> Union[list, None]:
        node_list = []
        
        # AGEグラフデータベースから指定タイプのエンティティを取得
        type_conditions = " OR ".join([f'properties(n).entity_type = "{t}"' for t in type_list])
        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity)
                     WHERE %s
                     RETURN properties(n).node_id AS node_id, properties(n) AS properties
                   $$) AS (node_id agtype, properties agtype)""" % (self.graph_name, type_conditions)
        
        try:
            records = await self._query(query)
            for record in records:
                if record and record.get("node_id"):
                    node_id = PGGraphStorage._decode_graph_label(record["node_id"])
                    node_list.append(node_id)
                    
            # 各ノードの詳細データを取得
            node_datas = await asyncio.gather(
                *[self.get_node(name) for name in node_list]
            )
            node_datas = [
                {**n, "entity_name": k}
                for k, n in zip(node_list, node_datas)
                if n is not None
            ]
            return node_datas
        except Exception as e:
            logger.error(f"Error getting nodes from types: {e}")
            return []
    
    # 修正: K-hopの近隣ノードを取得するメソッド（AGE対応）にして追加
    async def get_neighbors_within_k_hops(self, source_node_id: str, k):
        """
        AGEグラフデータベースでK-hopの近隣ノードを取得
        """
        if not await self.has_node(source_node_id):
            print("NO THIS ID:", source_node_id)
            return []
            
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        
        # AGEではCypherクエリでK-hopの近隣を取得
        query = """SELECT * FROM cypher('%s', $$
                     MATCH path = (start:Entity {node_id: "%s"})-[*1..%d]-(neighbor:Entity)
                     RETURN [n in nodes(path) | properties(n).node_id] AS path_nodes,
                            [r in relationships(path) | r] AS path_edges
                   $$) AS (path_nodes agtype, path_edges agtype)""" % (
            self.graph_name, src_label, k
        )
        
        try:
            records = await self._query(query)
            paths = []
            
            for record in records:
                if record and record.get("path_nodes"):
                    # ノードIDのパスをデコード
                    node_path = []
                    for node_id in record["path_nodes"]:
                        decoded_id = PGGraphStorage._decode_graph_label(node_id)
                        node_path.append(decoded_id)
                    
                    # エッジとして隣接ペアを作成
                    edges = []
                    for i in range(len(node_path) - 1):
                        edges.append((node_path[i], node_path[i + 1]))
                    paths.extend(edges)
            
            return paths
        except Exception as e:
            logger.error(f"Error getting neighbors within {k} hops: {e}")
            return []

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = PGGraphStorage._encode_graph_label(node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})
                     RETURN count(n) > 0 AS node_exists
                   $$) AS (node_exists bool)""" % (self.graph_name, entity_name_label)

        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["node_exists"],
        )

        return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))

        # 修正: エッジの向きを気にせずに「エッジがあるかどうか」をチェックしていたため、get_edge の処理と不整合が起きてエラーになっていた
        # has_edge の方も方向を指定するように修正
        # クエリの -[r]- を -[r]-> に変更
        query = """SELECT * FROM cypher('%s', $$
                    MATCH (a:Entity {node_id: "%s"})-[r]->(b:Entity {node_id: "%s"})
                    RETURN COUNT(r) > 0 AS edge_exists
                $$) AS (edge_exists bool)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )
        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["edge_exists"],
        )
        return single_result["edge_exists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})
                     RETURN n
                   $$) AS (n agtype)""" % (self.graph_name, label)
        record = await self._query(query)
        if record:
            node = record[0]
            node_dict = node["n"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query,
                node_dict,
            )
            return node_dict
        return None

    async def node_degree(self, node_id: str) -> int:
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})-[]->(x)
                     RETURN count(x) AS total_edge_count
                   $$) AS (total_edge_count integer)""" % (self.graph_name, label)
        record = (await self._query(query))[0]
        if record:
            edge_count = int(record["total_edge_count"])
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_count,
            )
            return edge_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            "{%s}:query:src_Degree+trg_degree:result:{%s}",
            inspect.currentframe().f_code.co_name,
            degrees,
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_id (str): Label of the source nodes
            target_node_id (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))

        # 修正: プロパティが空のエッジを除外して取得する
        query = """SELECT * FROM cypher('%s', $$
                     MATCH (a:Entity {node_id: \"%s\"})-[r]->(b:Entity {node_id: \"%s\"})
                     WHERE size(keys(properties(r))) > 0
                     RETURN properties(r) AS edge_properties
                     LIMIT 1
                   $$) AS (edge_properties agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )
        record = await self._query(query)
        if record and record[0] and record[0]["edge_properties"]:
            result = record[0]["edge_properties"]
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                result,
            )
            return result

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                      MATCH (n:Entity {node_id: "%s"})
                      OPTIONAL MATCH (n)-[r]-(connected)
                      RETURN n, r, connected
                    $$) AS (n agtype, r agtype, connected agtype)""" % (
            self.graph_name,
            label,
        )

        results = await self._query(query)
        edges = []
        for record in results:
            source_node = record["n"] if record["n"] else None
            connected_node = record["connected"] if record["connected"] else None

            source_label = (
                source_node["node_id"]
                if source_node and source_node["node_id"]
                else None
            )
            target_label = (
                connected_node["node_id"]
                if connected_node and connected_node["node_id"]
                else None
            )

            if source_label and target_label:
                edges.append(
                    (
                        PGGraphStorage._decode_graph_label(source_label),
                        PGGraphStorage._decode_graph_label(target_label),
                    )
                )

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the AGE database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
        properties = node_data

        query = """SELECT * FROM cypher('%s', $$
                     MERGE (n:Entity {node_id: "%s"})
                     SET n += %s
                     RETURN n
                   $$) AS (n agtype)""" % (
            self.graph_name,
            label,
            PGGraphStorage._format_properties(properties),
        )

        try:
            await self._query(query, readonly=False, upsert=True)
            logger.debug(
                "Upserted node with label '{%s}' and properties: {%s}",
                label,
                properties,
            )
        except Exception as e:
            logger.error("Error during upsert: {%s}", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))
        edge_properties = edge_data

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (source:Entity {node_id: "%s"})
                     WITH source
                     MATCH (target:Entity {node_id: "%s"})
                     MERGE (source)-[r:DIRECTED]->(target)
                     SET r += %s
                     RETURN r
                   $$) AS (r agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
            PGGraphStorage._format_properties(edge_properties),
        )
        # logger.info(f"-- inserting edge after formatted: {params}")
        try:
            await self._query(query, readonly=False, upsert=True)
            logger.debug(
                "Upserted edge from '{%s}' to '{%s}' with properties: {%s}",
                src_label,
                tgt_label,
                edge_properties,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def delete_by_chunk_ids(self, chunk_ids: list[str]):
        """指定されたチャンク ID 群を参照するノード・エッジを削除し、削除したエンティティ名とエッジ対を返す。

        Args:
            chunk_ids: LIGHTRAG_DOC_CHUNKS.id に相当するチャンク ID

        Returns:
            tuple[ list[str], list[tuple[str,str]] ] :
                削除したエンティティ名（node_id デコード済み）と、削除したエッジ (src,tgt) ペア。
        """
        if not chunk_ids:
            return [], []

        deleted_entities: set[str] = set()
        deleted_edge_pairs: set[tuple[str, str]] = set()

        for cid in chunk_ids:
            # --- 1) 対象ノードを取得 ---
            query_nodes = (
                """SELECT * FROM cypher('%s', $$
                   MATCH (n:Entity)
                   WHERE properties(n).source_id CONTAINS "%s"
                   RETURN properties(n).node_id AS node_id
                 $$) AS (node_id agtype)"""
                % (self.graph_name, cid)
            )
            try:
                node_records = await self._query(query_nodes)
                for rec in node_records:
                    node_enc = rec.get("node_id")
                    if node_enc:
                        node_dec = PGGraphStorage._decode_graph_label(node_enc)
                        deleted_entities.add(node_dec)
            except Exception as e:
                logger.error(f"Error searching nodes for chunk {cid}: {e}")

            # --- 2) 対象エッジを取得 ---
            query_edges = (
                """SELECT * FROM cypher('%s', $$
                   MATCH (a:Entity)-[r]->(b:Entity)
                   WHERE properties(r).source_id CONTAINS "%s"
                   RETURN properties(a).node_id AS src, properties(b).node_id AS tgt
                 $$) AS (src agtype, tgt agtype)"""
                % (self.graph_name, cid)
            )
            try:
                edge_records = await self._query(query_edges)
                for rec in edge_records:
                    src_enc = rec.get("src")
                    tgt_enc = rec.get("tgt")
                    if src_enc and tgt_enc:
                        src_dec = PGGraphStorage._decode_graph_label(src_enc)
                        tgt_dec = PGGraphStorage._decode_graph_label(tgt_enc)
                        deleted_edge_pairs.add(tuple(sorted((src_dec, tgt_dec))))
            except Exception as e:
                logger.error(f"Error searching edges for chunk {cid}: {e}")

            # --- 3) エッジ削除（source_id が該当チャンクを含むもの） ---
            del_edge_query = (
                """SELECT * FROM cypher('%s', $$
                   MATCH ()-[r]->() WHERE properties(r).source_id CONTAINS "%s" DELETE r
                 $$) AS (ignored agtype)"""
                % (self.graph_name, cid)
            )
            try:
                await self._query(del_edge_query, readonly=False)
            except Exception as e:
                logger.error(f"Error deleting edges for chunk {cid}: {e}")

            # --- 4) ノード削除（該当チャンクを参照するノード） ---
            del_node_query = (
                """SELECT * FROM cypher('%s', $$
                   MATCH (n:Entity) WHERE properties(n).source_id CONTAINS "%s" DETACH DELETE n
                 $$) AS (ignored agtype)"""
                % (self.graph_name, cid)
            )
            try:
                await self._query(del_node_query, readonly=False)
            except Exception as e:
                logger.error(f"Error deleting nodes for chunk {cid}: {e}")

        if deleted_entities:
            print(f"🗑️  Deleted {len(deleted_entities)} nodes referencing chunks")
        if deleted_edge_pairs:
            print(f"🗑️  Deleted {len(deleted_edge_pairs)} edges referencing chunks")

        return list(deleted_entities), list(deleted_edge_pairs)

    async def delete_node(self, node_id: str):
        """Delete a node and its connected edges from AGE graph."""
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
        query = (
            """SELECT * FROM cypher('%s', $$
               MATCH (n:Entity {node_id: \"%s\"}) DETACH DELETE n
             $$) AS (ignored agtype)"""
            % (self.graph_name, label)
        )
        try:
            await self._query(query, readonly=False)
            logger.info("Deleted node %s (and attached edges) from AGE", node_id)
        except Exception as e:
            logger.error("Error deleting node %s: %s", node_id, e)


NAMESPACE_TABLE_MAP = {
    "full_docs": "LIGHTRAG_DOC_FULL",
    "text_chunks": "LIGHTRAG_DOC_CHUNKS",
    "chunks": "LIGHTRAG_DOC_CHUNKS",
    "entities": "LIGHTRAG_VDB_ENTITY",
    "relationships": "LIGHTRAG_VDB_RELATION",
    "doc_status": "LIGHTRAG_DOC_STATUS",
    "llm_response_cache": "LIGHTRAG_LLM_CACHE",
}


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(255),
                    content TEXT,
                    content_vector VECTOR,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(256),
                    target_id VARCHAR(256),
                    content TEXT,
                    content_vector VECTOR,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
	                mode varchar(32) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, mode, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
                   metadata JSONB,
	               created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
	              )"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, metadata, created_at, updated_at
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode=$2
                               """,
    "get_by_mode_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                           FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode=$2 AND id=$3
                          """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id IN ({ids})
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, metadata, created_at, updated_at
                                   FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode= IN ({ids})
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_doc_full": """INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace, metadata)
                        VALUES ($1, $2, $3, $4::jsonb)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, updated_at = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,mode)
                                      VALUES ($1, $2, $3, $4, $5)
                                      ON CONFLICT (workspace,mode,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      mode=EXCLUDED.mode,
                                      updated_at = CURRENT_TIMESTAMP
                                     """,
    "upsert_chunk": """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, metadata)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      metadata=EXCLUDED.metadata,
                      updated_at = CURRENT_TIMESTAMP
                     """,
    "upsert_entity": """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content, content_vector, metadata)
                      VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      metadata=EXCLUDED.metadata,
                      updated_at=CURRENT_TIMESTAMP
                     """,
    "upsert_relationship": """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, metadata)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      metadata=EXCLUDED.metadata,
                      updated_at = CURRENT_TIMESTAMP
                     """,
    # SQL for VectorStorage
    "entities": """SELECT entity_name, distance, id, content FROM
        (SELECT workspace, id, entity_name, content, metadata, created_at, updated_at,
                1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
         FROM LIGHTRAG_VDB_ENTITY) AS subquery
        WHERE {where_clause}
       """,
    "entities_name": """SELECT entity_name, distance, id, content FROM
        (SELECT workspace, id, entity_name, content, metadata, created_at, updated_at,
                1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
         FROM LIGHTRAG_VDB_ENTITY) AS subquery
        WHERE {where_clause}
       """,
    "relationships": """SELECT source_id AS src_id, target_id AS tgt_id, content, distance FROM
        (SELECT workspace, id, source_id, target_id, content, metadata, created_at, updated_at,
                1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
         FROM LIGHTRAG_VDB_RELATION) AS subquery
        WHERE {where_clause}
       """,
    "chunks": """SELECT id, content, distance FROM
        (SELECT workspace, id, content, metadata, created_at, updated_at,
                1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
         FROM LIGHTRAG_DOC_CHUNKS) AS subquery
        WHERE {where_clause}
       """,
}