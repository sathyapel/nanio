# nanio

A production-grade, lightweight, modular TypeScript agentic framework featuring structured logging, cost-budget enforcement, resilient provider models (circuit breakers, rate limiting, full jitter retries), pluggable database persistence (PostgreSQL/pgvector, MongoDB, Qdrant REST), and tool integration.

## 📦 Package Architecture

`nanio` is organized as a monorepo containing the following packages under `/packages`:

*   **`@nanio/core`**: Core model interfaces (`BaseModel`, `BaseMemory`), Pg-backed chat memory, and config repositories.
*   **`@nanio/observability`**: Structured JSON logging (with automatic PII/secret scrubbing), request context propagation using `AsyncLocalStorage`, and pluggable performance telemetry / cost-budget trackers (Postgres, Mongo, Redis, Memory).
*   **`@nanio/providers`**: Resilient clients for Gemini, OpenAI, Claude, and xAI (Grok Chat & Image generation) implementing token buckets, circuit breakers, and exponential backoff retry policies.
*   **`@nanio/embeddings`**: Standard and batch-optimized embeddings (Gemini batch embedding).
*   **`@nanio/vectorstore`**: Pluggable similarity searches (Memory, MongoDB, PgVector, and Qdrant REST).
*   **`@nanio/registry`**: Model provider routers with fallback mechanisms.
*   **`@nanio/tools`**: Executable function schemas.

## 🚀 Getting Started

### Installation from NPM

You can install the modular `@nanio` packages directly from the NPM registry using the `@alpha` tag:

```bash
# Core package and observability layer
npm install @nanio/core@alpha @nanio/observability@alpha

# Provider clients, embeddings, and vector stores
npm install @nanio/providers@alpha @nanio/embeddings@alpha @nanio/vectorstore@alpha

# Registry and tools
npm install @nanio/registry@alpha @nanio/tools@alpha
```

### Local Workspace Development

If you are developing locally inside this monorepo:

1. Install dependencies at the root of the workspace:
   ```bash
   npm install
   ```

2. Compile all TypeScript packages:
   ```bash
   npm run build
   ```

3. Run the integrated Postgres/Qdrant/Model verification example:
   ```bash
   node packages/examples/dist/main.js
   ```

