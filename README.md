# nanio

> [!WARNING]
> **Status: Alpha (Experimental)** — `nanio` is currently in active, experimental development. API surfaces are subject to change.

🤝 **Collaborators & Contributors Needed!** We are actively looking for developers to join the project, contribute to the core architecture, add new database drivers, write providers, and refine agentic design patterns. Feel free to open issues or submit PRs!

A production-grade, lightweight, and modular TypeScript agentic framework designed to bring structured logging, safety limits, and high reliability to LLM application engineering.

## ✨ Key Features

*   **Dynamic LLM Routing & Registry (`@nanio/registry`)**: Dynamically route LLM completions across multiple models and endpoints with custom router mapping, fallback lists, and tier-specific overrides.
*   **IndexHydratedRAG (`@nanio/ihr`)**: Tree-structured RAG architecture implementing local `@zvec/zvec` collections for zero-server semantic indexing, pluggable embedding providers (Gemini, OpenAI, or local SentenceTransformers), contextual link expansion, hierarchical pruning, unconditional TF-IDF cosine re-ranking (with bigrams), LLM-powered multi-lingual auto-stopword detection, and recursive ancestor tree climbing.
*   **Pluggable Embeddings (`@nanio/embeddings`)**: Interchangeable `BaseEmbeddings` providers — **Gemini** (768-dim cloud), **OpenAI** (1536/3072-dim cloud), and **TransformersEmbeddings** (384-dim local ONNX, no API key). All three work identically in IHR and ZVecVectorStore.
*   **Minimal & Modular Tools (`@nanio/tools`)**: Standardized, lightweight function schemas powered by Zod validation, making it easy to expose custom capabilities to LLMs.
*   **Resilient AI Providers (`@nanio/providers`)**: Native REST clients for **Gemini, OpenAI, Claude, and xAI (Grok)** equipped with token-bucket rate limiters (supporting VIP tiers), circuit breakers to isolate downstream failures, and exponential retries with jitter.
*   **Pluggable Vector Stores (`@nanio/vectorstore`)**: Similarity search via **ZVecVectorStore** (local HNSW, no server), **PgVector**, **MongoDB**, and **Qdrant REST** — all accept any `BaseEmbeddings` provider.
*   **Structured JSON Observability (`@nanio/observability`)**: Seamless request context propagation using Node's `AsyncLocalStorage`, automatic PII/secret scrubbing (regex-based sanitization for emails, phone numbers, and keys), structured latency profiling via `timeOperation`, and execution cost budgets.


## 📦 Package Architecture

`nanio` is organized as a monorepo containing the following packages under `/packages`:

*   **`@nanio/core`**: Core model interfaces (`BaseModel`, `BaseMemory`), Pg-backed chat memory, and config repositories.
*   **`@nanio/observability`**: Structured JSON logging (with automatic PII/secret scrubbing), request context propagation using `AsyncLocalStorage`, `timeOperation` latency profiler, and pluggable performance telemetry / cost-budget trackers (Postgres, Mongo, Redis, Memory).
*   **`@nanio/providers`**: Resilient clients for Gemini, OpenAI, Claude, and xAI (Grok Chat & Image generation) implementing token buckets, circuit breakers, and exponential backoff retry policies.
*   **`@nanio/embeddings`**: Three interchangeable `BaseEmbeddings` providers — `GeminiEmbeddings` (768-dim, cloud), `OpenAIEmbeddings` (1536-dim, cloud), `TransformersEmbeddings` (384-dim, local ONNX). Swap providers in any consumer with zero other changes.
*   **`@nanio/vectorstore`**: Pluggable similarity searches — `ZVecVectorStore` (local HNSW, zero-server), `MemoryVectorStore`, `MongoVectorStore`, `PgVectorStore`, `QdrantVectorStore`. All accept any `BaseEmbeddings`.
*   **`@nanio/registry`**: Model provider routers with fallback mechanisms.
*   **`@nanio/tools`**: Executable function schemas.
*   **`@nanio/ihr`**: IndexHydratedRAG implementation using embedded Zvec vector search, pluggable `BaseEmbeddings`, relational SQL tables, and structured observability.

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

### 2. Pluggable Embedding Providers

All three providers share the same `BaseEmbeddings` interface and are interchangeable in any consumer:

```typescript
import { GeminiEmbeddings, OpenAIEmbeddings, TransformersEmbeddings } from '@nanio/embeddings';

// Cloud: Google Gemini — 768-dimensional vectors (batch API)
const gemini = new GeminiEmbeddings('gemini-embedding-001', { apiKey: process.env.GEMINI_API_KEY });

// Cloud: OpenAI — 1536-dimensional vectors (text-embedding-3-small)
//                 3072-dimensional vectors (text-embedding-3-large)
const openai = new OpenAIEmbeddings('text-embedding-3-small', { apiKey: process.env.OPENAI_API_KEY });

// Local: SentenceTransformers via ONNX Runtime — 384-dimensional, no API key needed
const local = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');

// Any of the three can be passed to ZVecVectorStore, IHR, PgVectorStore, etc.
const embedding = await local.embedQuery('TypeScript agentic framework');
console.log(`Dimension: ${embedding.length}`); // 384
```

### 3. ZVec Vector Store — Chat RAG with Any Provider

`ZVecVectorStore` uses local HNSW indexing (no server required). Pass any `BaseEmbeddings` provider and the dimension is auto-detected:

```typescript
import { ZVecVectorStore } from '@nanio/vectorstore';
import { GeminiEmbeddings, TransformersEmbeddings } from '@nanio/embeddings';

// --- Option A: ZVec + GeminiEmbeddings (cloud, 768-dim) ---
const geminiEmbeddings = new GeminiEmbeddings('gemini-embedding-001');
const geminiStore = new ZVecVectorStore(geminiEmbeddings, './zvec_data/chat_gemini', 'chat_kb');

await geminiStore.addDocuments([
  { pageContent: 'Nanio supports GeminiEmbeddings for cloud-scale retrieval.', metadata: { provider: 'gemini' } },
  { pageContent: 'ZVecVectorStore stores HNSW indices locally on disk.', metadata: { provider: 'zvec' } }
]);

const results = await geminiStore.similaritySearch('cloud semantic search', 2);
console.log(results[0].pageContent);

// --- Option B: ZVec + TransformersEmbeddings (local, 384-dim, no API key) ---
const localEmbeddings = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');
const localStore = new ZVecVectorStore(localEmbeddings, './zvec_data/chat_local', 'chat_local_kb');

await localStore.addDocuments([
  { pageContent: 'Local embeddings run fully offline with no API key.', metadata: { provider: 'transformers' } }
]);

const localResults = await localStore.similaritySearch('offline embedding', 1);
console.log(localResults[0].pageContent);
```

### 4. IndexHydratedRAG (IHR) with Embedded Zvec Index

IHR accepts **any `BaseEmbeddings` provider** — swap Gemini for local Transformers or OpenAI with no other code changes. The `embeddingProvider` field is emitted in all structured log lines:

```typescript
import pg from 'pg';
import { GeminiModel } from '@nanio/providers';
import { GeminiEmbeddings, TransformersEmbeddings } from '@nanio/embeddings';
import { IndexHydratedRAG, IngestSection } from '@nanio/ihr';

const db = new pg.Client('postgresql://localhost:5432/nanio');
await db.connect();

// --- Option A: Cloud provider (Gemini, 768-dim) ---
const cloudEmbeddings = new GeminiEmbeddings('gemini-embedding-001');

// --- Option B: Local provider (SentenceTransformers, 384-dim, no API key) ---
// const cloudEmbeddings = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');

const model = new GeminiModel('gemini-2.0-flash');
const ihr = new IndexHydratedRAG(db, cloudEmbeddings, model);

// Log shows: "embeddingProvider": "gemini-embedding-001"
console.log('Provider:', ihr.embeddingProviderName);

await ihr.initializeSchema();

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

// Retrieve: runs subtree pruning, LLM-powered auto-stopword detection, TF-IDF cosine re-ranking, and ancestor climbing
const response = await ihr.retrieve('Explain the core architecture.', 'doc_123', {
  autoStopwords: true // Automatically detects the corpus language via LLM and generates custom stopwords
});

console.log('Answer:', response.answer);
console.log('Context Tree Lineage:', response.context);
console.log('Path used:', response.path); // Always 'tfidf'
```

### 5. IndexHydratedRAG (IHR) Use Cases

IndexHydratedRAG is designed for complex document hierarchies where section relationships (parent/sibling/ancestor outlines) are critical for LLM understanding:

*   **Technical Outlines & Documentation Manuals**: When querying highly nested structural outlines (e.g. manuals with Section 1.2.1 under 1.2). Ancestor climbing reconstructs the outline path so the LLM retains full architectural context.
*   **Multi-Lingual Publications & Books**: LLM-powered automatic stopword generation detects the primary language of the retrieved corpus (e.g. French, Spanish, German) dynamically, generating language-specific stopwords on the fly to yield precise keyword matching for TF-IDF re-ranking.
*   **Legal Contracts & Compliance Documents**: Navigating clauses, sub-clauses, and cross-references. Subtree pruning deduplicates overlapping parent-child text hierarchies to fit context budgets while keeping the logical document sequence intact.
*   **Medical Diagnostic Manuals**: Querying clinical structures where keyword alignment is paramount. Unconditional TF-IDF cosine re-ranking uses a vocabulary of unigrams and bigrams fitted across candidates to elevate exact matches.
*   **Hierarchical Course Curricula & Textbooks**: Accessing educational materials organized by unit, chapter, and sub-chapter. For student queries targeting highly specific exercises or formulas, ancestor climbing supplies the parent chapter's baseline theory to ground the LLM's explanation.

## 🚀 Getting Started

### Installation from NPM

You can install the modular `@nanio` packages directly from the NPM registry using the `@alpha` tag:

```bash
# Core package and observability layer
npm install @nanio/core@alpha @nanio/observability@alpha

# Provider clients, embeddings, and vector stores
npm install @nanio/providers@alpha @nanio/embeddings@alpha @nanio/vectorstore@alpha

# IndexHydratedRAG, Registry, and Tools
npm install @nanio/ihr@alpha @nanio/registry@alpha @nanio/tools@alpha
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

3. Run the integrated verification example (Postgres mock, Qdrant mock, ZVec, IHR, multi-provider):
   ```bash
   node packages/examples/dist/main.js
   ```

