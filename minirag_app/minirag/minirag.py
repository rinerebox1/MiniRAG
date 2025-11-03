import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Any, Iterable, Optional, Sequence
from dotenv import load_dotenv


from .operate import (
    chunking_by_token_size,
    extract_entities,
    hybrid_query,
    minirag_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    clean_text,
    get_content_summary,
    set_logger,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    DocStatus,
)


STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.jsondocstatus_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorge": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "WeaviateVectorStorage": ".kg.weaviate_impl",
    "WeaviateKVStorage": ".kg.weaviate_impl",
    "WeaviateGraphStorage": ".kg.weaviate_impl",
    "run_sync": ".kg.weaviate_impl",
}

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

load_dotenv(dotenv_path=".env", override=False)


@dataclass
class _InsertPayload:
    documents: list[str]
    ids: Optional[list[str]]
    metadatas: Optional[list[dict[str, Any]]]
    structured_records: list[dict[str, Any]]


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class MiniRAG:
    working_dir: str = field(
        default_factory=lambda: f"./minirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # RAGmode: str = 'minirag'

    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    
    # 🆕 multi-field search settings
    enable_field_splitting: bool = True  # フィールド分割の有効化
    generate_combined_chunk: bool = True  # 統合版チャンク(_all)の生成
    text_field_keys: list[str] = field(
        default_factory=lambda: ["title", "description", "summary", "content", "body", "text"]
    )  # 自動的にテキストフィールドとして扱うキー

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = None
    llm_model_name: str = (
        "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    )
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Add new field for document status storage type
    doc_status_storage: str = field(default="JsonDocStatusStorage")

    # Custom Chunking Function
    chunking_func: callable = chunking_by_token_size
    chunking_func_kwargs: dict = field(default_factory=dict)

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "minirag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"MiniRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )

        self.key_string_value_json_storage_cls = partial(
            self.key_string_value_json_storage_cls, global_config=global_config
        )

        self.vector_db_storage_cls = partial(
            self.vector_db_storage_cls, global_config=global_config
        )

        self.graph_storage_cls = partial(
            self.graph_storage_cls, global_config=global_config
        )
        self.json_doc_status_storage = self.key_string_value_json_storage_cls(
            namespace="json_doc_status_storage",
            embedding_func=None,
        )

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        global_config = asdict(self)

        self.entity_name_vdb = self.vector_db_storage_cls(
            namespace="entities_name",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )

        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )
        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)
        self.doc_status = self.doc_status_storage_cls(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=None,
        )

    def _get_storage_class(self, storage_name: str) -> dict:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def set_storage_client(self, db_client):
        # Now only tested on Oracle Database
        self.storage_client = db_client
        for storage in [
            self.vector_db_storage_cls,
            self.graph_storage_cls,
            self.doc_status,
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.key_string_value_json_storage_cls,
            self.chunks_vdb,
            self.relationships_vdb,
            self.entities_vdb,
            self.entity_name_vdb,    # 修正: 追加
            self.graph_storage_cls,
            self.chunk_entity_relation_graph,
            self.llm_response_cache,
        ]:
            # set client
            storage.db = db_client

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(
        self,
        input: str | dict | list[str] | list[dict],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        metadatas: dict | list[dict] | None = None,
        overwrite: bool = False,
        schema: dict | None = None,
        text_fields: Sequence[str] | None = None,
    ) -> None:
        payload = self._prepare_insert_payload(
            input=input,
            ids=ids,
            metadatas=metadatas,
            schema=schema,
            text_fields=text_fields,
        )

        print(f"🚀 AINSERT called with overwrite={overwrite}")
        print(f"📥 Input: {len(payload.documents)} documents")
        print(f"📥 IDs: {payload.ids}")
        print(f"📥 Metadatas: {payload.metadatas}")

        await self.apipeline_enqueue_documents(
            payload.documents,
            payload.ids,
            payload.metadatas,
            overwrite,
        )
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

        if payload.structured_records and schema and self._is_postgres_backend():
            await self._write_structured_records_to_pg(
                payload.structured_records,
                schema,
            )

        await self._insert_done()

    def _prepare_insert_payload(
        self,
        input: str | dict | list[str] | list[dict],
        ids: str | list[str] | None,
        metadatas: dict | list[dict] | None,
        schema: dict | None,
        text_fields: Sequence[str] | None,
    ):
        is_structured_input = self._looks_like_structured_records(input)

        if not is_structured_input:
            documents = self._ensure_string_list(input)
            ids_list = self._normalize_ids_argument(ids, len(documents))
            metadata_list = self._normalize_metadata_argument(metadatas, len(documents))
            if metadata_list and ids_list is None:
                raise ValueError(
                    "Explicit IDs are required when providing metadatas"
                )
            return _InsertPayload(
                documents=documents,
                ids=ids_list,
                metadatas=metadata_list,
                structured_records=[],
            )

        records = self._ensure_record_list(input)
        schema = schema or {}
        id_column = schema.get("id_column", "doc_id")
        field_specs = schema.get("fields", {})

        override_ids = self._normalize_ids_argument(ids, len(records))
        override_metadatas = self._normalize_metadata_argument(
            metadatas,
            len(records),
        )

        documents: list[str] = []
        resolved_ids: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        structured_records: list[dict[str, Any]] = []

        for idx, record in enumerate(records):
            record_copy = dict(record)
            record_metadata = record_copy.pop("metadata", {}) or {}

            structured_values = self._apply_schema_to_record(
                record_copy,
                field_specs,
                id_column,
            )

            text_segments = self._collect_text_segments(
                record_copy,
                structured_values,
                text_fields,
                field_specs,
            )
            if not text_segments:
                raise ValueError(
                    "No text fields found for structured record insertion"
                )

            doc_text = "\n".join(segment for segment in text_segments if segment).strip()
            documents.append(doc_text)

            candidate_id = structured_values.get(id_column) or record_copy.get(id_column)
            if candidate_id is None:
                candidate_id = compute_mdhash_id(doc_text, prefix="doc-")
            resolved_ids.append(str(candidate_id))

            merged_metadata: dict[str, Any] = {}
            merged_metadata.update(record_metadata)
            for field_name, value in structured_values.items():
                if field_name == id_column:
                    continue
                merged_metadata[field_name] = value

            if override_metadatas and override_metadatas[idx]:
                merged_metadata.update(override_metadatas[idx])
            
            # 🆕 フィールド分割用に元のレコードを保存
            if self.enable_field_splitting:
                merged_metadata["_original_data"] = record

            metadata_list.append(merged_metadata)

            structured_record = {
                field_name: structured_values.get(field_name)
                for field_name in field_specs.keys()
            }
            if id_column not in structured_record:
                structured_record[id_column] = candidate_id
            structured_records.append(structured_record)

        ids_list = override_ids or resolved_ids
        return _InsertPayload(
            documents=documents,
            ids=ids_list,
            metadatas=metadata_list,
            structured_records=structured_records,
        )

    @staticmethod
    def _looks_like_structured_records(input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return True
        if isinstance(input_data, list) and input_data:
            return all(isinstance(item, dict) for item in input_data)
        return False

    @staticmethod
    def _ensure_string_list(input_data: str | list[str]) -> list[str]:
        if isinstance(input_data, str):
            return [input_data]
        return list(input_data)

    @staticmethod
    def _ensure_record_list(
        input_data: dict | list[dict],
    ) -> list[dict[str, Any]]:
        if isinstance(input_data, dict):
            return [dict(input_data)]
        return [dict(record) for record in input_data]

    @staticmethod
    def _normalize_ids_argument(
        ids: str | list[str] | None,
        expected_length: int,
    ) -> Optional[list[str]]:
        if ids is None:
            return None
        if isinstance(ids, str):
            if expected_length != 1:
                raise ValueError("Number of IDs must match the number of documents")
            return [ids]
        if len(ids) != expected_length:
            raise ValueError("Number of IDs must match the number of documents")
        return [str(doc_id) for doc_id in ids]

    @staticmethod
    def _normalize_metadata_argument(
        metadatas: dict | list[dict] | None,
        expected_length: int,
    ) -> Optional[list[dict[str, Any]]]:
        if metadatas is None:
            return None
        if isinstance(metadatas, dict):
            return [dict(metadatas) for _ in range(expected_length)]
        if len(metadatas) != expected_length:
            raise ValueError(
                "Number of metadatas must match the number of documents"
            )
        return [dict(metadata or {}) for metadata in metadatas]

    def _apply_schema_to_record(
        self,
        record: dict[str, Any],
        field_specs: dict[str, dict],
        id_column: str,
    ) -> dict[str, Any]:
        if not field_specs:
            return dict(record)

        structured: dict[str, Any] = {}
        for field_name, spec in field_specs.items():
            value = record.get(field_name)
            if value is None:
                if spec.get("nullable", True) is False:
                    raise ValueError(
                        f"Field '{field_name}' is not nullable but value is None"
                    )
                structured[field_name] = None
                continue
            structured[field_name] = self._coerce_field_value(value, spec)
        return structured

    def _collect_text_segments(
        self,
        record: dict[str, Any],
        structured_values: dict[str, Any],
        text_fields: Sequence[str] | None,
        field_specs: dict[str, dict],
    ) -> list[str]:
        if text_fields:
            candidates = list(text_fields)
        elif field_specs:
            candidates = [
                field
                for field, spec in field_specs.items()
                if spec.get("type", "").lower() in {"text", "varchar", "character varying"}
            ]
        else:
            candidates = [
                field
                for field, value in record.items()
                if isinstance(value, str)
            ]

        text_segments: list[str] = []
        for field in candidates:
            value = record.get(field, structured_values.get(field))
            if value is None:
                continue
            if isinstance(value, list):
                text_segments.extend(str(item) for item in value if item is not None)
            else:
                text_segments.append(str(value))
        return text_segments

    @staticmethod
    def _coerce_field_value(value: Any, spec: dict[str, Any]) -> Any:
        field_type = (spec.get("type") or "").lower()
        if field_type in {"text", "varchar", "character varying"}:
            if isinstance(value, list):
                return "\n".join(str(item) for item in value if item is not None)
            return str(value)
        if field_type in {"integer", "int", "int4", "bigint", "smallint"}:
            return None if value is None else int(value)
        if field_type in {"float", "double", "double precision", "real"}:
            return None if value is None else float(value)
        if field_type in {"numeric", "decimal"}:
            return None if value is None else float(value)
        if field_type in {"boolean", "bool"}:
            return None if value is None else bool(value)
        # timestampや日付などは呼び出し元で正しい型に変換すると仮定
        return value

    def _is_postgres_backend(self) -> bool:
        return self.doc_status_storage == "PGDocStatusStorage"

    async def _write_structured_records_to_pg(
        self,
        records: Iterable[dict[str, Any]],
        schema: dict,
    ) -> None:
        table_name = schema.get("table")
        if not table_name:
            raise ValueError("`schema['table']` is required for structured inserts")

        field_specs = schema.get("fields")
        if not field_specs:
            raise ValueError("`schema['fields']` must define target columns")

        db_client = getattr(self.doc_status, "db", None)
        if db_client is None:
            logger.warning("PostgreSQL client is not configured; skipping structured insert")
            return

        columns = list(field_specs.keys())
        placeholders = ", ".join(f"${idx + 1}" for idx in range(len(columns)))

        conflict_columns = schema.get("conflict_columns")
        upsert_clause = ""
        if conflict_columns:
            conflict_list = ", ".join(conflict_columns)
            update_columns = [col for col in columns if col not in conflict_columns]
            set_clause = ", ".join(
                f"{col} = EXCLUDED.{col}"
                for col in update_columns
            )
            upsert_clause = f" ON CONFLICT ({conflict_list}) DO UPDATE SET {set_clause}" if set_clause else " ON CONFLICT ({conflict_list}) DO NOTHING"

        insert_sql = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({placeholders}){upsert_clause}"
        )

        for record in records:
            params = {column: record.get(column) for column in columns}
            await db_client.execute(insert_sql, params)

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(metadatas, dict):
            metadatas = [metadatas]

        if ids is not None:
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")
            if metadatas and len(metadatas) != len(input):
                raise ValueError("Number of metadatas must match the number of documents")
            contents = {id_: doc for id_, doc in zip(ids, input)}
            if metadatas:
                contents_with_meta = {
                    id_: (doc, meta)
                    for id_, doc, meta in zip(ids, input, metadatas)
                }

        else:
            input = list(set(clean_text(doc) for doc in input))
            contents = {compute_mdhash_id(doc, prefix="doc-"): doc for doc in input}

        if metadatas:
            contents_with_meta = {
                id_: (doc, meta) for id_, doc, meta in zip(ids, input, metadatas)
            }
        else:
            contents_with_meta = {}

        unique_contents = {
            id_: content
            for content, id_ in {
                content: id_ for id_, content in contents.items()
            }.items()
        }
        new_docs: dict[str, Any] = {
            id_: {
                "content": content,
                "content_summary": get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "metadata": contents_with_meta.get(id_, (None, {}))[1]
                if metadatas
                else {},
            }
            for id_, content in unique_contents.items()
        }

        all_new_doc_ids = set(new_docs.keys())
        
        if overwrite:
            # 上書きモード：重複チェックをスキップして全てのドキュメントを処理
            unique_new_doc_ids = all_new_doc_ids
            logger.info(f"Overwrite mode: Processing all {len(unique_new_doc_ids)} documents")
        else:
            # 通常モード：重複ドキュメントを除外
            unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
            logger.info(f"Normal mode: Processing {len(unique_new_doc_ids)} new documents (filtered from {len(all_new_doc_ids)})")

        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }
        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        if overwrite and new_docs:
            # 上書きモードの場合、既存のチャンクを削除
            print(f"🔥 OVERWRITE MODE: Deleting existing chunks for {len(new_docs)} documents")
            await self._delete_existing_chunks(list(new_docs.keys()))
        
        # デバッグ：保存するドキュメントのメタデータを確認
        for doc_id, doc_data in new_docs.items():
            print(f"📝 Storing doc '{doc_id}' with metadata: {doc_data.get('metadata', {})}")
        
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} documents (overwrite={overwrite})")

    async def _delete_existing_chunks(self, doc_ids: list[str]) -> None:
        """上書きモード用：指定されたドキュメントIDに関連する既存データをカスケード削除"""
        try:
            # Step 1: Retrieve chunk IDs associated with the document IDs
            chunk_ids = await self.text_chunks.get_chunk_ids_by_doc_ids(doc_ids)
            if not chunk_ids:
                logger.info(f"No chunks found for doc_ids: {doc_ids}. Skipping deletion.")
                return
            
            # Step 2: Delete nodes and edges from the knowledge graph
            deleted_entities, deleted_edge_pairs = await self.chunk_entity_relation_graph.delete_by_chunk_ids(chunk_ids)

            # Step 3: Delete related vectors
            ent_vec_ids = [compute_mdhash_id(e, prefix="ent-") for e in deleted_entities]
            ename_vec_ids = [compute_mdhash_id(e, prefix="Ename-") for e in deleted_entities]
            rel_vec_ids = [compute_mdhash_id(src + tgt, prefix="rel-") for src, tgt in deleted_edge_pairs]

            delete_tasks = [
                self.entities_vdb.delete_by_ids(ent_vec_ids),
                self.entity_name_vdb.delete_by_ids(ename_vec_ids),
                self.relationships_vdb.delete_by_ids(rel_vec_ids),
                self.chunks_vdb.delete_by_doc_ids(doc_ids),
                self.text_chunks.delete_by_doc_ids(doc_ids)
            ]

            await asyncio.gather(*delete_tasks)

            logger.info(
                f"Cascade delete successful for {len(doc_ids)} documents: "
                f"{len(chunk_ids)} chunks, {len(deleted_entities)} entities, "
                f"and {len(deleted_edge_pairs)} relationships deleted."
            )
        except Exception as e:
            logger.error(f"Failed to cascade delete existing data: {e}", exc_info=True)
            # Decide if you want to proceed or re-raise
            # For now, we log the error and proceed, which was the previous behavior.
            logger.warning("Proceeding with upsert despite cascade delete failure.")

    def _normalize_metadata(self, metadata: Any) -> dict:
        """
        メタデータを辞書に正規化する
        
        Args:
            metadata: メタデータ（辞書、JSON文字列、Noneのいずれか）
        
        Returns:
            dict: 正規化されたメタデータ（常に辞書）
        """
        if isinstance(metadata, dict):
            return metadata
        elif isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse metadata as JSON: {e}. Using empty dict.")
                return {}
        else:
            return {}
    
    def _extract_text_fields(self, data: dict) -> tuple[dict[str, str], dict]:
        """
        dictからテキストフィールドとメタデータを分離
        
        Args:
            data: 入力データ（例: {"doc_id": "...", "title": "...", "description": [...], "metadata": {...}}）
        
        Returns:
            text_fields: テキストフィールドの辞書（例: {"title": "...", "description": "..."}）
            metadata: その他のフィールド（数値など）とmetadataをマージしたもの
        """
        text_fields = {}
        metadata = data.get("metadata", {}).copy()
        
        for key, value in data.items():
            # doc_id と metadata はスキップ
            if key in ["doc_id", "metadata"]:
                continue
            
            # text_field_keys に含まれる、またはstr/listの値はテキストフィールドとして扱う
            if key in self.text_field_keys or isinstance(value, (str, list)):
                # リストの場合は改行で結合
                if isinstance(value, list):
                    text_fields[key] = "\n".join(str(v) for v in value if v)
                else:
                    text_fields[key] = str(value)
            else:
                # 数値などはmetadataへ
                metadata[key] = value
        
        return text_fields, metadata
    
    def _generate_chunks_per_field(
        self,
        doc_id: str,
        text_fields: dict[str, str],
        base_metadata: dict
    ) -> dict[str, dict]:
        """
        フィールドごとにチャンクを生成
        
        Args:
            doc_id: ドキュメントID
            text_fields: テキストフィールドの辞書（例: {"title": "...", "description": "..."}）
            base_metadata: ベースとなるメタデータ
        
        Returns:
            全チャンクの辞書（chunk_id -> chunk_data）
        """
        all_chunks = {}
        
        # フィールドごとのチャンク生成
        for field_name, field_content in text_fields.items():
            if not field_content or not field_content.strip():
                continue
                
            field_chunks = self.chunking_func(
                field_content,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name
            )
            
            for chunk in field_chunks:
                chunk_id = compute_mdhash_id(
                    chunk["content"] + field_name + doc_id,
                    prefix=f"chunk-{field_name}-"
                )
                all_chunks[chunk_id] = {
                    **chunk,
                    "full_doc_id": doc_id,
                    "metadata": {
                        **base_metadata,
                        "text_field": field_name  # 🆕 フィールド識別子
                    }
                }
        
        # 統合版チャンク生成（デフォルト検索用）
        if self.generate_combined_chunk:
            combined_content = "\n".join(text_fields.values())
            combined_chunks = self.chunking_func(
                combined_content,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name
            )
            
            for chunk in combined_chunks:
                chunk_id = compute_mdhash_id(
                    chunk["content"] + "_all" + doc_id,
                    prefix="chunk-all-"
                )
                all_chunks[chunk_id] = {
                    **chunk,
                    "full_doc_id": doc_id,
                    "metadata": {
                        **base_metadata,
                        "text_field": "_all"  # 🆕 統合版マーカー
                    }
                }
        
        return all_chunks

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.
        """
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )

        to_process_docs: dict[str, Any] = {
            **processing_docs,
            **failed_docs,
            **pending_docs,
        }
        if not to_process_docs:
            logger.info("No documents to process")
            return

        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]
        logger.info(f"Number of batches to process: {len(docs_batches)}")

        for batch_idx, docs_batch in enumerate(docs_batches):
            for doc_id, status_doc in docs_batch:
                print(f"⚙️  Processing doc '{doc_id}', status_doc.metadata = {status_doc.metadata}")
                
                # メタデータを辞書に正規化（文字列の場合はJSONパース、Noneの場合は空辞書）
                metadata = self._normalize_metadata(status_doc.metadata)
                # status_doc.metadata を更新（後続処理で使用されるため）
                status_doc.metadata = metadata
                
                # 🆕 フィールド分割が有効な場合はフィールドごとにチャンクを生成
                if self.enable_field_splitting and metadata:
                    # status_doc.content を解析してフィールド分割できるか試みる
                    # メタデータに元の構造化データがある場合を想定
                    original_data = metadata.get("_original_data")
                    
                    if original_data and isinstance(original_data, dict):
                        # 構造化データからフィールド分割
                        text_fields, merged_metadata = self._extract_text_fields(original_data)
                        chunks = self._generate_chunks_per_field(doc_id, text_fields, merged_metadata)
                        print(f"📦 Created {len(chunks)} field-split chunks for doc '{doc_id}' (fields: {list(text_fields.keys())})")
                    else:
                        # 通常のチャンク生成（後方互換性）
                        chunks = {
                            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                **dp,
                                "full_doc_id": doc_id,
                                "metadata": {**metadata, "text_field": "_all"},  # 🆕 _all マーカー
                            }
                            for dp in self.chunking_func(
                                status_doc.content,
                                self.chunk_overlap_token_size,
                                self.chunk_token_size,
                                self.tiktoken_model_name,
                            )
                        }
                        print(f"📦 Created {len(chunks)} standard chunks for doc '{doc_id}'")
                else:
                    # フィールド分割無効時は通常のチャンク生成
                    chunks = {
                        compute_mdhash_id(dp["content"], prefix="chunk-"): {
                            **dp,
                            "full_doc_id": doc_id,
                            "metadata": {**metadata, "text_field": "_all"},  # 🆕 _all マーカー
                        }
                        for dp in self.chunking_func(
                            status_doc.content,
                            self.chunk_overlap_token_size,
                            self.chunk_token_size,
                            self.tiktoken_model_name,
                        )
                    }
                    print(f"📦 Created {len(chunks)} standard chunks for doc '{doc_id}'")
                
                # 各チャンクのメタデータを確認
                for chunk_id, chunk_data in list(chunks.items())[:2]:  # 最初の2つだけ表示
                    text_field = chunk_data.get('metadata', {}).get('text_field', 'N/A')
                    print(f"   └─ Chunk '{chunk_id[:16]}...' text_field: {text_field}, metadata: {chunk_data.get('metadata', {})}")
                await asyncio.gather(
                    self.chunks_vdb.upsert(chunks),
                    self.full_docs.upsert(
                        {doc_id: {"content": status_doc.content, "metadata": metadata}}
                    ),
                    self.text_chunks.upsert(chunks),
                )

                if chunks:
                    logger.info(f"Performing entity extraction on {len(chunks)} chunks for doc '{doc_id}'")
                    await extract_entities(
                        chunks,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entity_vdb=self.entities_vdb,
                        entity_name_vdb=self.entity_name_vdb,
                        relationships_vdb=self.relationships_vdb,
                        global_config=asdict(self),
                    )

                await self.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSED,
                            "chunks_count": len(chunks),
                            "content": status_doc.content,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    }
                )
        logger.info("Document processing pipeline completed")

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def _apply_target_fields_filter(self, param: QueryParam) -> QueryParam:
        """
        target_fields を metadata_filter に変換
        
        Args:
            param: 元の QueryParam
        
        Returns:
            変換された QueryParam
        """
        field_filter = {}
        
        if param.target_fields is None:
            # デフォルト: 統合検索
            field_filter = {"text_field": "_all"}
        elif len(param.target_fields) == 1:
            # 単一フィールド検索
            field_filter = {"text_field": param.target_fields[0]}
        else:
            # 複数フィールド検索（リストとして渡す → postgres_impl.py で IN 句に変換）
            field_filter = {"text_field": param.target_fields}
        
        # 既存のmetadata_filterとマージ
        if param.metadata_filter:
            merged_filter = {**param.metadata_filter, **field_filter}
        else:
            merged_filter = field_filter
        
        # 新しいQueryParamを作成（datac copyを使ってimmutableに）
        from dataclasses import replace
        return replace(param, metadata_filter=merged_filter)
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        # 🆕 target_fields を metadata_filter に変換
        if param.target_fields is not None:
            param = self._apply_target_fields_filter(param)
        
        if param.mode == "light":
            response, source = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                self.chunks_vdb,
                param,
                asdict(self),
            )
        elif param.mode == "mini":
            print("★デバッグ(minirag.py): mini mode")
            response, source = await minirag_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.entity_name_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                self.embedding_func,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response, source = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response, source

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            if hasattr(self.entity_name_vdb, 'delete_entity'):
                await self.entity_name_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
