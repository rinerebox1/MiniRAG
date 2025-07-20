## Prerequisites

Before using PostgreSQL with MiniRAG, ensure you have the following prerequisites met:

### PostgreSQL Server
*   **Version**: MiniRAG is generally compatible with modern PostgreSQL versions. While a specific minimum version is not explicitly stated in the source, PostgreSQL 12+ is recommended for broad feature support, including partitioning and other performance enhancements that can be beneficial. For Apache AGE and pgvector, refer to their respective documentation for PostgreSQL version compatibility.
*   **pgvector Extension**: For vector similarity search capabilities (used in `PGVectorStorage`), the `pgvector` extension must be installed in your PostgreSQL database. You can typically install it using `CREATE EXTENSION IF NOT EXISTS vector;` if the extension is available on your database server.
*   **Apache AGE Extension**: For graph database functionalities (used in `PGGraphStorage`), the Apache AGE (A Graph Extension) must be installed and configured in your PostgreSQL database. Follow the official Apache AGE installation guide for your PostgreSQL version.

### Python Dependencies
Ensure the following Python libraries are installed in your MiniRAG environment. MiniRAG attempts to install `asyncpg` if it's missing, but it's good practice to manage dependencies explicitly.
*   `asyncpg`: For asynchronous interaction with PostgreSQL.
*   `psycopg-pool`: For connection pooling (mentioned in test files, good for robust applications).
*   `psycopg[binary,pool]`: Alternative PostgreSQL adapter with binary and pooling support (mentioned in test files).

You can typically install these using pip:
```bash
pip install asyncpg psycopg-pool "psycopg[binary,pool]"
```

## Configuration

To connect MiniRAG to your PostgreSQL instance, you need to provide connection parameters. This is typically done when initializing MiniRAG components that require database interaction.

The primary class managing PostgreSQL connections is `PostgreSQLDB`. It expects a configuration dictionary with the following keys:

*   `host` (str): The hostname or IP address of your PostgreSQL server. Default: `"localhost"`.
*   `port` (int): The port number on which PostgreSQL is listening. Default: `5432`.
*   `user` (str): The PostgreSQL username. **Required.**
*   `password` (str): The password for the specified user. **Required.**
*   `database` (str): The name of the PostgreSQL database to connect to. **Required.**
*   `workspace` (str): A namespace within the MiniRAG tables to isolate data for different projects or instances. Default: `"default"`. This allows you to use the same database for multiple MiniRAG setups without data collision. All table entries created by MiniRAG will be associated with this workspace identifier.

### Example Configuration

Here's an example of how you might structure the configuration dictionary in Python:

```python
postgres_config = {
    "host": "your_postgres_host",
    "port": 5432,
    "user": "your_postgres_user",
    "password": "your_postgres_password",
    "database": "minirag_db",
    "workspace": "my_project_space"
}

# This config would then be passed to MiniRAG components, for example:
# from minirag.kg.postgres_impl import PostgreSQLDB
# db_instance = PostgreSQLDB(config=postgres_config)
# (Further details on component initialization will be covered in "Usage with MiniRAG")
```

**Note:** Ensure that the specified `user` has the necessary permissions (CREATE, SELECT, INSERT, UPDATE, DELETE) on the target `database` and for creating extensions if they are not already installed (though it's often better to have extensions installed beforehand by a superuser).

## Initialization and Table Creation

When MiniRAG initializes a connection to PostgreSQL (e.g., when `PostgreSQLDB.initdb()` is called), it automatically checks for the existence of required tables. If the tables are not found, MiniRAG will attempt to create them.

The DDL (Data Definition Language) for these tables is defined internally within `minirag/kg/postgres_impl.py`. The tables created by MiniRAG include:

*   `LIGHTRAG_DOC_FULL`: Stores the full content of ingested documents.
*   `LIGHTRAG_DOC_CHUNKS`: Stores processed text chunks from documents, potentially with their vector embeddings (if using `pgvector`).
*   `LIGHTRAG_VDB_ENTITY`: Stores vector embeddings for entities, used for semantic search.
*   `LIGHTRAG_VDB_RELATION`: Stores vector embeddings for relationships between entities.
*   `LIGHTRAG_LLM_CACHE`: Caches responses from Language Models (LLMs) to avoid redundant API calls.
*   `LIGHTRAG_DOC_STATUS`: Tracks the processing status of documents ingested into MiniRAG.

All tables are created within the public schema by default (unless your PostgreSQL user or search path is configured differently) and include a `workspace` column, which is used to isolate data as specified in your connection configuration.

While MiniRAG handles table creation, it's crucial that the configured PostgreSQL `user` has the necessary permissions to create tables in the specified `database`.

## Apache AGE (A Graph Extension) Setup

MiniRAG leverages Apache AGE for graph storage and querying capabilities, implemented in the `PGGraphStorage` class.

### Prerequisites
*   **Apache AGE Installation**: As mentioned in the prerequisites, Apache AGE must be installed and enabled in your PostgreSQL instance. Refer to the [official Apache AGE documentation](https://age.apache.org/docs/current/intro/installation/) for installation instructions.
*   **Database Configuration**: Ensure AGE is properly loaded and configured in your `postgresql.conf` (e.g., by adding `age` to `shared_preload_libraries` and restarting PostgreSQL).

### Graph Creation and Configuration
*   **Graph Existence**: The `PGGraphStorage` component expects a graph to be present in the database. While the `postgres_impl_test.py` shows examples of `create_graph`, the production `PGGraphStorage` class itself doesn't automatically create the graph. It's generally assumed the graph specified by `AGE_GRAPH_NAME` already exists or is created by a setup script.
    *   You can create a graph using a PGSQL client connected to your database with AGE enabled:
        ```sql
        LOAD 'age';
        SET search_path = ag_catalog, "$user", public;
        SELECT create_graph('your_graph_name');
        ```
*   **`AGE_GRAPH_NAME` Environment Variable**: The name of the graph to be used by MiniRAG's `PGGraphStorage` is specified through the `AGE_GRAPH_NAME` environment variable. You must set this environment variable in the environment where your MiniRAG application runs.
    ```bash
    export AGE_GRAPH_NAME="my_minirag_graph"
    ```
    Or set it within your Python application before `PGGraphStorage` is initialized:
    ```python
    import os
    os.environ["AGE_GRAPH_NAME"] = "my_minirag_graph"
    ```
*   **Search Path**: The `PGGraphStorage` implementation automatically sets the `search_path` to `ag_catalog, "$user", public` for its operations to ensure AGE functions and objects are correctly resolved.

### Usage
When `PGGraphStorage` is initialized and used, it will perform Cypher queries against the specified graph within PostgreSQL. Ensure the PostgreSQL `user` configured for MiniRAG has the necessary permissions to operate on the AGE graph (e.g., query, create/update nodes and edges).

## Usage with MiniRAG

Once your PostgreSQL server is set up with the necessary extensions (`pgvector`, Apache AGE) and you have your connection configuration ready, you can instruct MiniRAG to use PostgreSQL for its various storage backends.

MiniRAG's architecture allows for different storage mechanisms for different types of data:
*   **Key-Value Storage (`BaseKVStorage`)**: Used for storing document full text (`full_docs`), text chunks (`text_chunks`), and LLM response cache (`llm_response_cache`). Implemented by `PGKVStorage`.
*   **Vector Storage (`BaseVectorStorage`)**: Used for storing vector embeddings of chunks, entities, and relationships to enable semantic search. Implemented by `PGVectorStorage`.
*   **Document Status Storage (`DocStatusStorage`)**: Used for tracking the processing status of documents. Implemented by `PGDocStatusStorage`.
*   **Graph Storage (`BaseGraphStorage`)**: Used for storing and querying knowledge graphs. Implemented by `PGGraphStorage`.

### Initializing MiniRAG with PostgreSQL

You typically specify the use of PostgreSQL when initializing the main `MiniRAG` class or its underlying components. The configuration provided will be used to instantiate the PostgreSQL-backed storage classes.

Here's a conceptual example of how you might initialize MiniRAG components to use PostgreSQL. Note that the exact initialization might vary based on how you're structuring your MiniRAG application (e.g., using `MiniRAG.from_config()` or manually composing components).

```python
import asyncio
from minirag.minirag import MiniRAG
from minirag.utils import Config
from minirag.kg.postgres_impl import (
    PostgreSQLDB,
    PGKVStorage,
    PGVectorStorage,
    PGDocStatusStorage,
    PGGraphStorage
)

# 1. Define your PostgreSQL connection configuration
postgres_config_dict = {
    "host": "localhost",
    "port": 5432,
    "user": "your_user",
    "password": "your_password",
    "database": "minirag_db",
    "workspace": "my_project_workspace"
}

# 2. (If using Apache AGE) Set the graph name environment variable
import os
os.environ["AGE_GRAPH_NAME"] = "my_minirag_graph"

# 3. Create a global MiniRAG config (can also be loaded from a YAML file)
#    This example focuses on setting storage implementation classes.
#    In a full setup, you'd also configure LLMs, embedding models, etc.
global_config_dict = {
    "version": "0.1.0",
    "llm_api_key": "your_llm_api_key", # Example, replace with actual config
    "embedding_model_name": "your_embedding_model", # Example
    "embedding_batch_num": 32,
    # Specify PostgreSQL implementations for storage
    "kv_storage_cls": "minirag.kg.postgres_impl.PGKVStorage",
    "vector_storage_cls": "minirag.kg.postgres_impl.PGVectorStorage",
    "doc_status_storage_cls": "minirag.kg.postgres_impl.PGDocStatusStorage",
    "graph_storage_cls": "minirag.kg.postgres_impl.PGGraphStorage",
    # Central PostgreSQL database configuration
    "postgres_db_config": postgres_config_dict,
    # Arguments for specific storage classes (db instance will be passed by MiniRAG)
    "vector_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    # Add other *_cls_kwargs if they need non-default parameters other than 'db'
    # "kv_storage_cls_kwargs": {},
    # "doc_status_storage_cls_kwargs": {},
    # "graph_storage_cls_kwargs": {},
}

async def main():
    # Create Config object
    config = Config(config_dict=global_config_dict)

    # MiniRAG.from_config(config) will handle PostgreSQLDB initialization using "postgres_db_config"
    # and pass the db instance to the storage components.

    # Example: Manually initializing a KV storage (illustrative, as MiniRAG handles this)
    # Note: The `namespace` and `embedding_func` are illustrative for storage classes
    # and are usually set by the parent component (e.g., Indexer, Querier).
    # A PostgreSQLDB instance `pg_db` would be created by MiniRAG and passed.

    # kv_store_fulldocs = PGKVStorage(namespace="full_docs", global_config=config, embedding_func=None, db=rag_instance.postgres_db)
    # vector_store_chunks = PGVectorStorage(namespace="chunks", global_config=config, embedding_func=my_embedding_function, db=rag_instance.postgres_db)
    # doc_status_store = PGDocStatusStorage(global_config=config, db=rag_instance.postgres_db)
    # graph_store = PGGraphStorage(namespace="my_graph_namespace", global_config=config, embedding_func=None, db=rag_instance.postgres_db)


    # Typically, you'd initialize MiniRAG which then sets up these storages
    # based on the configuration.
    # rag_instance = MiniRAG.from_config(config=config)
    # await rag_instance.init_async() # This would initialize the DB connection among other things

    # After initialization, you can use MiniRAG as usual for indexing and querying.
    # For example (conceptual):
    # await rag_instance.indexer.run_pipeline_async(docs_data)
    # results = await rag_instance.querier.query_async("What is MiniRAG?")
    # print(results)

    print("MiniRAG components (conceptually) initialized with PostgreSQL backend.")
    print(f"Ensure your PostgreSQL server is running at {postgres_config_dict['host']}:{postgres_config_dict['port']}")
    print(f"and the database '{postgres_config_dict['database']}' and user '{postgres_config_dict['user']}' are correctly set up.")
    print(f"For graph operations, Apache AGE should be active on graph '{os.getenv('AGE_GRAPH_NAME')}'.")

    # Close the pool when done (important in a real application)
    # If using rag_instance:
    # if rag_instance.postgres_db and rag_instance.postgres_db.pool:
    #     await rag_instance.postgres_db.pool.close()

if __name__ == "__main__":
    # Define a dummy embedding function for the example if needed by PGVectorStorage
    # async def my_embedding_function(texts: list[str]):
    #     # Replace with your actual embedding model call
    #     print(f"Embedding {len(texts)} texts (dummy).")
    #     # Example: return list of dummy vectors of appropriate dimension
    #     return [[0.1] * 768 for _ in texts]

    asyncio.run(main())
```

This example demonstrates:
1.  Defining the PostgreSQL connection dictionary.
2.  Setting the `AGE_GRAPH_NAME` environment variable.
3.  Creating a `Config` object where you specify a central `postgres_db_config` for the PostgreSQL connection. MiniRAG will use this to initialize a `PostgreSQLDB` instance and pass it to the specified PostgreSQL-backed storage classes (`PGKVStorage`, `PGVectorStorage`, etc.). You also specify any other necessary arguments for these classes in their respective `*_cls_kwargs` sections.
4.  A `main` async function to illustrate initialization. In a complete MiniRAG application, `MiniRAG.from_config()` would typically handle the instantiation of these storage backends based on the provided configuration.

By configuring MiniRAG this way, all data persistence and retrieval operations for the selected components will be handled by your PostgreSQL database.
