# @nanio/ihr

IndexHydratedRAG (IHR) tree-structured document ingestion and retrieval engine for the `nanio` agentic framework.

## Key Features

*   **Relational Tree Layout**: Relational `documents` and `section_table` in Postgres to represent document parent-child structures.
*   **In-Process Vector Search**: Embedded HNSW vector indexing using local `@zvec/zvec` collections.
*   **Context Link Expansion**: Fetches related and sibling section IDs via Zvec metadata.
*   **Tree Pruning**: Removes redundant nodes when parent and child are both fetched.
*   **Unconditional TF-IDF Cosine Re-rank**: Normalised TF-IDF vectors using unigrams and bigrams to score candidate sections.
*   **LLM Auto-Stopwords**: LLM-powered multi-lingual auto-stopword detection dynamically fetching stopwords based on detected text language.
*   **Recursive Outline Climbing**: Climbs section levels to reconstruct structured heading paths for the LLM.
