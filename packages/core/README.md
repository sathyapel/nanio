# @nanio/core

Core interfaces, models, and repositories for the `nanio` agentic framework.

## Key Concepts

*   **`BaseModel`**: Abstract class for all LLM providers (Gemini, Claude, OpenAI, Grok).
*   **`Message`**: Interface representing chat roles and contents.
*   **`ChatConfig`**: Class defining per-user tiers, budget metrics, and model configurations.
*   **`PgChatMemory`**: Postgres-backed persistent session storage for conversation turns.
*   **`PgLLMConfigRepository`**: Admin dashboard config repositories for loading settings per user tier.
