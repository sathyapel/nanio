# @nanio/embeddings

Pluggable and interchangeable embedding clients for the `nanio` agentic framework.

## Key Features

*   **`BaseEmbeddings`**: Single unified interface to switch providers seamlessly in vector stores and RAG pipelines.
*   **`GeminiEmbeddings`**: Cloud-based Google Gemini embeddings supporting automatic batching (up to 100 documents per request).
*   **`OpenAIEmbeddings`**: Cloud-based OpenAI embeddings supporting text-embedding-3-small and text-embedding-3-large models.
*   **`TransformersEmbeddings`**: Fully offline local embeddings using `@huggingface/transformers` (Xenova/all-MiniLM-L6-v2), running entirely in-process without requiring an API key.
