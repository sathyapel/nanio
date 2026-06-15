# @nanio/vectorstore

Pluggable similarity search vector stores for the `nanio` agentic framework.

## Key Features

*   **`BaseVectorStore`**: Shared standard vector database interface.
*   **`ZVecVectorStore`**: Zero-server in-process local HNSW vector database powered by `@zvec/zvec` (Alibaba's fast index).
*   **`PgVectorStore`**: Postgres `pgvector`-backed vector store using standard cosine distance operators (`<=>`).
*   **`QdrantVectorStore`**: Native Qdrant client communicating over REST fetch APIs without bulky SDKs.
*   **`MongoVectorStore`**: Persistent MongoDB atlas search collections.
