# nanio

> [!WARNING]
> **Status: Alpha (Experimental)** — `nanio` is currently in active, experimental development. API surfaces are subject to change.

🤝 **Collaborators & Contributors Needed!** We are actively looking for developers to join the project, contribute to the core architecture, add new database drivers, write providers, and refine agentic design patterns. Feel free to open issues or submit PRs!

A production-grade, lightweight, and modular TypeScript agentic framework designed to bring structured logging, safety limits, and high reliability to LLM application engineering.

## ✨ Key Features

*   **Dynamic LLM Routing & Registry (`@nanio/registry`)**: Dynamically route LLM completions across multiple models and endpoints with custom router mapping, fallback lists, and tier-specific overrides.
*   **IndexHydratedRAG (`@nanio/ihr`)**: Tree-structured RAG architecture implementing local `@zvec/zvec` collections for zero-server semantic indexing, local `@huggingface/transformers` feature extraction, contextual link expansion, hierarchical pruning, context token gating, and recursive ancestor tree climbing.
*   **Minimal & Modular Tools (`@nanio/tools`)**: Standardized, lightweight function schemas powered by Zod validation, making it easy to expose custom capabilities to LLMs.
*   **Resilient AI Providers (`@nanio/providers`)**: Native REST clients for **Gemini, OpenAI, Claude, and xAI (Grok)** equipped with token-bucket rate limiters (supporting VIP tiers), circuit breakers to isolate downstream failures, and exponential retries with jitter.
*   **Pluggable Database Persistence**: Drivers for similarity searching (**PgVector, MongoDB, and Qdrant REST**), PostgreSQL-backed session memory repositories, cost configurations, and metric counters.
*   **Structured JSON Observability (`@nanio/observability`)**: Seamless request context propagation using Node's `AsyncLocalStorage`, automatic PII/secret scrubbing (regex-based sanitization for emails, phone numbers, and keys), and execution budgets.


## 📦 Package Architecture

`nanio` is organized as a monorepo containing the following packages under `/packages`:

*   **`@nanio/core`**: Core model interfaces (`BaseModel`, `BaseMemory`), Pg-backed chat memory, and config repositories.
*   **`@nanio/observability`**: Structured JSON logging (with automatic PII/secret scrubbing), request context propagation using `AsyncLocalStorage`, and pluggable performance telemetry / cost-budget trackers (Postgres, Mongo, Redis, Memory).
*   **`@nanio/providers`**: Resilient clients for Gemini, OpenAI, Claude, and xAI (Grok Chat & Image generation) implementing token buckets, circuit breakers, and exponential backoff retry policies.
*   **`@nanio/embeddings`**: Standard and batch-optimized embeddings (Gemini batch embedding, local Transformers embeddings).
*   **`@nanio/vectorstore`**: Pluggable similarity searches (Memory, MongoDB, PgVector, and Qdrant REST).
*   **`@nanio/registry`**: Model provider routers with fallback mechanisms.
*   **`@nanio/tools`**: Executable function schemas.
*   **`@nanio/ihr`**: IndexHydratedRAG implementation using embedded Zvec vector search and relational SQL tables.

## 💡 Quick Examples

### 1. Dynamic LLM Registry Routing

Here is a quick example showing how to initialize clients, wrap them in a dynamic fallback `LLMRegistry`, and invoke the generation with tier-based `ChatConfig` overrides:

```typescript
import { GeminiModel, ClaudeModel } from '@nanio/providers';
import { LLMRegistry } from '@nanio/registry';
import { ChatConfig } from '@nanio/core';

// 1. Initialize different model clients
const primary = new GeminiModel('gemini-2.0-flash');
const fallback = new ClaudeModel('claude-3-5-sonnet-20241022');

// 2. Set up the registry fallback chain
const registry = new LLMRegistry([
  { tier: 'PRIMARY', model: primary, name: 'gemini-primary' },
  { tier: 'LIGHTER', model: fallback, name: 'claude-fallback' }
]);

// 3. Configure generation settings with tier overrides
const config = new ChatConfig({
  userId: 'user_12345',
  userTier: 'FREE', // Enforces FREE tier rate-limit rules
  modelTier: 'PRIMARY', // Starts attempts at the PRIMARY tier
  temperature: 0.85
});

// 4. Generate completions (will automatically failover to Claude if Gemini hits rate limits)
const response = await registry.generate(
  [{ role: 'user', content: 'Design a lightweight TypeScript architecture.' }],
  { config }
);

console.log(response.content);
```

### 2. IndexHydratedRAG (IHR) with Embedded Zvec Index

Here is an example showing how to ingest a document outline tree and perform IndexHydratedRAG queries using local Zvec vector storage and local SentenceTransformer model embeddings:

```typescript
import pg from 'pg';
import { GeminiModel } from '@nanio/providers';
import { TransformersEmbeddings } from '@nanio/embeddings';
import { IndexHydratedRAG, IngestSection } from '@nanio/ihr';

// 1. Initialize DB client and local embeddings
const db = new pg.Client("postgresql://localhost:5432/nanio");
await db.connect();

const embeddings = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');
const model = new GeminiModel('gemini-2.0-flash');

const ihr = new IndexHydratedRAG(db, embeddings, model);
await ihr.initializeSchema();

// 2. Ingest structured document outline tree
const sections: IngestSection[] = [
  {
    section_id: '1',
    parent_id: null,
    level: 1,
    heading: 'Introduction to Nanio',
    content: '',
    summary: 'Overview of the nanio lightweight agentic framework.'
  },
  {
    section_id: '1.1',
    parent_id: '1',
    level: 2,
    heading: 'Core Architecture',
    content: 'The core architecture is based on lightweight ESM packages.',
    summary: 'Explains the ESM package structures and core module.'
  }
];

await ihr.ingest('doc_123', 'Nanio Specifications', 'https://nanio.dev/specs', sections);

// 3. Query the collection (runs fast path, TF-IDF gating, pruning, and ancestor climbing)
const response = await ihr.retrieve('Explain the core architecture.', 'doc_123');

console.log('Answer:', response.answer);
console.log('Context Tree Lineage:', response.context);
```

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

