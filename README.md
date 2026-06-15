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

### Installation

Install dependencies at the root of the workspace:

```bash
npm install
```

### Build

Compile all TypeScript packages:

```bash
npm run build
```

### Run Demo

Run the integrated Postgres/Qdrant/Model verification example:

```bash
node packages/examples/dist/main.js
```
