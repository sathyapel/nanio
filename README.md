# nanio

> [!WARNING]
> **Status: Alpha (Experimental)** — `nanio` is currently in active, experimental development. API surfaces are subject to change.

🤝 **Collaborators & Contributors Needed!** We are actively looking for developers to join the project, contribute to the core architecture, add new database drivers, write providers, and refine agentic design patterns. Feel free to open issues or submit PRs!

A production-grade, lightweight, and modular TypeScript agentic framework designed to bring structured logging, safety limits, and high reliability to LLM application engineering.

## ✨ Key Features

*   **Dynamic LLM Routing & Registry (`@nanio/registry`)**: Dynamically route LLM completions across multiple models and endpoints with custom router mapping, fallback lists, and tier-specific overrides.
*   **Minimal & Modular Tools (`@nanio/tools`)**: Standardized, lightweight function schemas powered by Zod validation, making it easy to expose custom capabilities to LLMs.
*   **Resilient AI Providers (`@nanio/providers`)**: Native REST clients for **Gemini, OpenAI, Claude, and xAI (Grok)** equipped with token-bucket rate limiters (supporting VIP tiers), circuit breakers to isolate downstream failures, and exponential retries with jitter.
*   **Pluggable Database Persistence**: Drivers for similarity searching (**PgVector, MongoDB, and Qdrant REST**), PostgreSQL-backed session memory repositories, cost configurations, and metric counters.
*   **Structured JSON Observability (`@nanio/observability`)**: Seamless request context propagation using Node's `AsyncLocalStorage`, automatic PII/secret scrubbing (regex-based sanitization for emails, phone numbers, and keys), and execution budgets.


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

