import * as dotenv from 'dotenv';
import fs from 'fs';
import { z } from 'zod';
import { 
  SimpleAgent, 
  ToolCallingAgent, 
  BaseModel, 
  Message, 
  LLMResponse, 
  GenerateOptions,
  PgLLMConfigRepository,
  PgChatMemory,
  ChatConfig
} from '@nanio/core';
import { FunctionTool } from '@nanio/tools';
import { GeminiModel, BaseAIClient, ClaudeModel, XAIModel, XAIImageClient } from '@nanio/providers';
import { GeminiEmbeddings, TransformersEmbeddings, OpenAIEmbeddings } from '@nanio/embeddings';
import { MemoryVectorStore, MongoVectorStore, PgVectorStore, QdrantVectorStore, ZVecVectorStore, Document } from '@nanio/vectorstore';
import { FallbackRouter, LLMRegistry } from '@nanio/registry';
import { IndexHydratedRAG, IngestSection } from '@nanio/ihr';
import {
  runWithContext,
  newRequestContext,
  getContext,
  setupLogging,
  getLogger,
  Events,
  hashArgs,
  CostTracker,
  MongoCostStore,
  PgCostStore,
  BudgetStatus,
  PerfTracker,
  MemoryPerfStore,
  PgPerfStore,
  timeOperation
} from '@nanio/observability';

// 1. Load environment variables
dotenv.config();

// 2. Initialize telemetry logging
setupLogging('INFO');
const logger = getLogger('MainIntegrationDemo');

// ============================================================================
// Cosine Similarity helper for Vector Stores Mocking
// ============================================================================
function dotProduct(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

function magnitude(a: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = dotProduct(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

// ============================================================================
// PostgreSQL In-Memory Mock Client
// ============================================================================
class MockPgClient {
  public tables: Record<string, any[]> = {
    documents: [],
    llm_cost_events: [],
    perf_latencies: [],
    perf_counters: [],
    perf_gauges: [],
    llm_configs: [],
    conversation_turns: [],
    section_table: [],
    index_table: []
  };

  private lastSerialIds: Record<string, number> = {
    documents: 0,
    perf_latencies: 0,
    conversation_turns: 0
  };

  async query(sql: string, params: any[] = []): Promise<{ rows: any[] }> {
    const sqlClean = sql.trim().replace(/\s+/g, ' ');

    if (sqlClean.startsWith('CREATE') || sqlClean.startsWith('ALTER') || sqlClean.startsWith('DROP')) {
      return { rows: [] };
    }

    if (sqlClean.includes('INSERT INTO documents')) {
      if (sqlClean.includes('source_url')) {
        const [id, title, source_url] = params;
        this.tables.documents.push({
          id,
          title,
          source_url,
          ingested_at: new Date()
        });
        return { rows: [] };
      } else {
        const content = params[0];
        const metadata = typeof params[1] === 'string' ? JSON.parse(params[1]) : params[1];
        const embedding = params[2].replace(/[\[\]]/g, '').split(',').map(Number);
        this.tables.documents.push({
          id: ++this.lastSerialIds.documents,
          content,
          metadata,
          embedding
        });
        return { rows: [] };
      }
    }

    if (sqlClean.includes('SELECT content, metadata, (embedding <=> $1)')) {
      const queryVector = params[0].replace(/[\[\]]/g, '').split(',').map(Number);
      const limit = params[1];

      const scored = this.tables.documents.map(doc => {
        const dist = 1.0 - cosineSimilarity(queryVector, doc.embedding);
        return {
          content: doc.content,
          metadata: doc.metadata,
          distance: dist
        };
      });

      scored.sort((a, b) => a.distance - b.distance);
      return { rows: scored.slice(0, limit) };
    }

    if (sqlClean.includes('DELETE FROM section_table')) {
      const docId = params[0];
      this.tables.section_table = this.tables.section_table.filter(s => s.doc_id !== docId);
      return { rows: [] };
    }

    if (sqlClean.includes('DELETE FROM index_table')) {
      const docId = params[0];
      this.tables.index_table = this.tables.index_table.filter(s => s.doc_id !== docId);
      return { rows: [] };
    }

    if (sqlClean.includes('INSERT INTO section_table')) {
      const [doc_id, section_id, parent_id, level, heading, content, summary] = params;
      this.tables.section_table.push({
        id: this.tables.section_table.length + 1,
        doc_id,
        section_id,
        parent_id,
        level,
        heading,
        content,
        summary
      });
      return { rows: [] };
    }

    if (sqlClean.includes('INSERT INTO index_table')) {
      const [section_id, doc_id, summary, embedding, related_ids, sibling_ids] = params;
      this.tables.index_table.push({
        section_id,
        doc_id,
        summary,
        embedding,
        related_ids,
        sibling_ids
      });
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT section_id, embedding FROM index_table') && !sqlClean.includes('ANY')) {
      const docId = params[0];
      const rows = this.tables.index_table.filter(r => r.doc_id === docId);
      return { rows };
    }

    if (sqlClean.includes('SELECT section_id, related_ids, sibling_ids FROM index_table') && sqlClean.includes('ANY')) {
      const docId = params[0];
      const sectionIds = params[1];
      const rows = this.tables.index_table.filter(r => r.doc_id === docId && sectionIds.includes(r.section_id));
      return { rows };
    }

    if (sqlClean.includes('SELECT section_id, length(content) AS chars FROM section_table')) {
      const docId = params[0];
      const sectionIds = params[1];
      const rows = this.tables.section_table
        .filter(r => r.doc_id === docId && sectionIds.includes(r.section_id))
        .map(r => ({
          section_id: r.section_id,
          chars: r.content ? r.content.length : 0
        }));
      return { rows };
    }

    if ((sqlClean.includes('SELECT section_id, heading, content FROM section_table') || sqlClean.includes('SELECT section_id, heading, summary, content FROM section_table')) && sqlClean.includes('ANY')) {
      const docId = params[0];
      const sectionIds = params[1];
      const rows = this.tables.section_table
        .filter(r => r.doc_id === docId && sectionIds.includes(r.section_id))
        .map(r => ({
          section_id: r.section_id,
          heading: r.heading,
          summary: r.summary,
          content: r.content
        }));
      return { rows };
    }

    if (sqlClean.includes('SELECT section_id, summary FROM index_table') && sqlClean.includes('ANY')) {
      const docId = params[0];
      const sectionIds = params[1];
      const rows = this.tables.index_table.filter(r => r.doc_id === docId && sectionIds.includes(r.section_id));
      return { rows };
    }

    if (sqlClean.includes('WITH RECURSIVE ancestors AS')) {
      const docId = params[0];
      const sectionIds = params[1];
      const results = new Map<string, any>();

      const findNode = (secId: string) => {
        return this.tables.section_table.find(n => n.doc_id === docId && n.section_id === secId);
      };

      const queue = [...sectionIds];
      while (queue.length > 0) {
        const currentId = queue.shift();
        if (!currentId) continue;
        if (results.has(currentId)) continue;

        const node = findNode(currentId);
        if (node) {
          results.set(currentId, {
            id: node.id,
            doc_id: node.doc_id,
            section_id: node.section_id,
            parent_id: node.parent_id,
            level: node.level,
            heading: node.heading,
            content: node.content
          });
          if (node.parent_id) {
            queue.push(node.parent_id);
          }
        }
      }

      const rows = Array.from(results.values());
      rows.sort((a, b) => {
        if (a.level !== b.level) return a.level - b.level;
        return a.section_id.localeCompare(b.section_id);
      });

      return { rows };
    }

    if (sqlClean.includes('INSERT INTO llm_cost_events')) {
      this.tables.llm_cost_events.push({
        id: Math.random().toString(),
        timestamp: new Date(),
        service: params[0],
        model: params[1],
        user_id: params[2],
        session_id: params[3],
        turn_id: params[4],
        input_tokens: params[5],
        output_tokens: params[6],
        cost_usd: parseFloat(params[7] || '0.0'),
        coin_revenue_usd: 0.0
      });
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT COALESCE(SUM(cost_usd), 0) as total FROM llm_cost_events')) {
      const timestampCutoff = params[0];
      const matching = this.tables.llm_cost_events.filter(e => e.timestamp >= timestampCutoff);
      const sum = matching.reduce((acc, curr) => acc + curr.cost_usd, 0);
      return { rows: [{ total: sum }] };
    }

    if (sqlClean.includes('SELECT COALESCE(SUM(CASE WHEN timestamp >= $2 THEN cost_usd ELSE 0 END), 0) as daily')) {
      const userId = params[0];
      const startOfToday = params[1];
      const startOfThisMonth = params[2];

      const dailyMatching = this.tables.llm_cost_events.filter(e => e.user_id === userId && e.timestamp >= startOfToday);
      const monthlyMatching = this.tables.llm_cost_events.filter(e => e.user_id === userId && e.timestamp >= startOfThisMonth);

      const dailySum = dailyMatching.reduce((acc, curr) => acc + curr.cost_usd, 0);
      const monthlySum = monthlyMatching.reduce((acc, curr) => acc + curr.cost_usd, 0);

      return {
        rows: [{
          daily: dailySum,
          monthly: monthlySum
        }]
      };
    }

    if (sqlClean.includes('SELECT COALESCE(SUM(cost_usd), 0) as total, COALESCE(SUM(CASE WHEN service = \'chat\'')) {
      const userId = params[0];
      const startDate = params[1];
      const endDate = params[2];

      const matching = this.tables.llm_cost_events.filter(e => e.user_id === userId && e.timestamp >= startDate && e.timestamp < endDate);

      const total = matching.reduce((acc, curr) => acc + curr.cost_usd, 0);
      const chat = matching.filter(e => e.service === 'chat').reduce((acc, curr) => acc + curr.cost_usd, 0);
      const live = matching.filter(e => e.service === 'live').reduce((acc, curr) => acc + curr.cost_usd, 0);
      const image = matching.filter(e => e.service === 'image').reduce((acc, curr) => acc + curr.cost_usd, 0);
      const embed = matching.filter(e => e.service === 'embed').reduce((acc, curr) => acc + curr.cost_usd, 0);
      const turns = matching.filter(e => e.service === 'chat').length;

      return {
        rows: [{
          total,
          chat,
          live,
          image,
          embed,
          turns
        }]
      };
    }

    if (sqlClean.includes('INSERT INTO perf_latencies')) {
      const metric = params[0];
      const latencyMs = params[1];
      this.tables.perf_latencies.push({
        id: ++this.lastSerialIds.perf_latencies,
        metric,
        latency_ms: latencyMs,
        timestamp: new Date()
      });
      return { rows: [] };
    }

    if (sqlClean.includes('DELETE FROM perf_latencies WHERE id IN')) {
      const metric = params[0];
      const maxSamples = params[1];

      const metricSamples = this.tables.perf_latencies.filter(s => s.metric === metric);
      metricSamples.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
      const keep = metricSamples.slice(0, maxSamples).map(s => s.id);

      this.tables.perf_latencies = this.tables.perf_latencies.filter(s => s.metric !== metric || keep.includes(s.id));
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT latency_ms FROM perf_latencies WHERE metric = $1')) {
      const metric = params[0];
      const samples = this.tables.perf_latencies.filter(s => s.metric === metric);
      return { rows: samples };
    }

    if (sqlClean.includes('INSERT INTO perf_counters') && sqlClean.includes('ON CONFLICT')) {
      const key = params[0];
      const amount = params[1];
      let row = this.tables.perf_counters.find(r => r.key === key);
      if (!row) {
        row = { key, value: 0 };
        this.tables.perf_counters.push(row);
      }
      row.value += amount;
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT value FROM perf_counters WHERE key = $1')) {
      const key = params[0];
      const row = this.tables.perf_counters.find(r => r.key === key);
      return { rows: row ? [row] : [] };
    }

    if (sqlClean.includes('INSERT INTO perf_gauges') && sqlClean.includes('ON CONFLICT')) {
      const key = params[0];
      const value = params[1];
      const expiresAt = params[2];
      let row = this.tables.perf_gauges.find(r => r.key === key);
      if (!row) {
        row = { key, value, expires_at: expiresAt };
        this.tables.perf_gauges.push(row);
      } else {
        row.value = value;
        row.expires_at = expiresAt;
      }
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT * FROM llm_configs WHERE user_tier = $1')) {
      const tier = params[0];
      const row = this.tables.llm_configs.find(r => r.user_tier === tier);
      return { rows: row ? [row] : [] };
    }

    if (sqlClean.includes('SELECT COALESCE(MAX(session_turn_index), 0) as max_idx FROM conversation_turns')) {
      const sessionId = params[0];
      const matching = this.tables.conversation_turns.filter(t => t.session_id === sessionId);
      const maxIdx = matching.reduce((max, curr) => Math.max(max, curr.session_turn_index), 0);
      return { rows: [{ max_idx: maxIdx }] };
    }

    if (sqlClean.includes('INSERT INTO conversation_turns')) {
      const id = ++this.lastSerialIds.conversation_turns;
      this.tables.conversation_turns.push({
        id,
        turn_id: params[0],
        session_id: params[1],
        user_id: params[2],
        session_turn_index: params[3],
        user_message: params[4],
        ai_response: params[5],
        stage_at_turn: params[6],
        created_at: new Date()
      });
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT id FROM conversation_turns WHERE session_id = $1 ORDER BY session_turn_index DESC LIMIT 1')) {
      const sessionId = params[0];
      const matching = this.tables.conversation_turns.filter(t => t.session_id === sessionId);
      matching.sort((a, b) => b.session_turn_index - a.session_turn_index);
      return { rows: matching.slice(0, 1) };
    }

    if (sqlClean.includes('UPDATE conversation_turns SET ai_response = $1 WHERE id = $2')) {
      const aiResponse = params[0];
      const id = params[1];
      const turn = this.tables.conversation_turns.find(t => t.id === id);
      if (turn) {
        turn.ai_response = aiResponse;
      }
      return { rows: [] };
    }

    if (sqlClean.includes('SELECT user_message, ai_response FROM conversation_turns WHERE session_id = $1 ORDER BY session_turn_index ASC')) {
      const sessionId = params[0];
      const matching = this.tables.conversation_turns.filter(t => t.session_id === sessionId);
      matching.sort((a, b) => a.session_turn_index - b.session_turn_index);
      return { rows: matching };
    }

    if (sqlClean.includes('DELETE FROM conversation_turns WHERE session_id = $1')) {
      const sessionId = params[0];
      this.tables.conversation_turns = this.tables.conversation_turns.filter(t => t.session_id !== sessionId);
      return { rows: [] };
    }

    throw new Error(`Unhandled mock SQL query: ${sqlClean}`);
  }
}

// ============================================================================
// Qdrant Mock REST interceptor
// ============================================================================
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const urlStr = typeof input === 'string'
    ? input
    : (input instanceof URL
      ? input.toString()
      : (input as Request).url);
  const url = new URL(urlStr);

  if (url.origin === 'http://localhost:6333') {
    const path = url.pathname;

    if (path.startsWith('/collections/') && path.endsWith('/points/search')) {
      return new Response(JSON.stringify({
        result: [
          {
            score: 0.98,
            payload: {
              pageContent: 'Nanio is a lightweight typescript agentic framework featuring structured JSON observability.',
              metadata: { category: 'docs' }
            }
          }
        ]
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    if (path.startsWith('/collections/')) {
      return new Response(JSON.stringify({ result: { status: 'ok' } }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  return originalFetch(urlStr, init);
};

// ============================================================================
// Main Execution
// ============================================================================
async function runDemo() {
  logger.info('=== INITIALIZING TELEMETRY & PERSISTENT SERVICES ===');

  // 1. Initialize Mock Pg Client
  const mockPg = new MockPgClient() as any;

  // 2. Initialize Pg-backed cost store and performance metric tracker
  const pgCostStore = new PgCostStore(mockPg);
  const costTracker = new CostTracker(pgCostStore);
  const pgPerfStore = new PgPerfStore(mockPg);
  const perfTracker = new PerfTracker(pgPerfStore);

  // Initialize DB tables
  await pgCostStore.initializeSchema();
  await pgPerfStore.initializeSchema();

  // 3. Setup Context variables
  const context = newRequestContext({
    userId: 'c52bde0e-ca22-4467-bc18-0a068e3e4a99',
    sessionId: '642bde0e-ca22-4467-bc18-0a068e3e4a99',
    stage: 'ENGAGED',
    tier: 'PRIMARY'
  });

  await runWithContext(context, async () => {
    logger.info('Application running inside PostgreSQL request context storage.');

    // 4. Verify context variable extraction
    const activeCtx = getContext();
    logger.info('Active request context parameters:', {
      traceId: activeCtx.traceId,
      userId: activeCtx.userId,
      tier: activeCtx.tier
    });

    // 5. Test dynamic budget check
    logger.info('Checking budget state before model execution...');
    const budgetState = await costTracker.checkBudget(activeCtx.userId, activeCtx.tier);
    logger.info(`Budget state is: ${budgetState}`);

    // 6. Test pricing and billing increments in Postgres
    logger.info('Simulating API billing increments in PostgreSQL cost store...');
    const billingCost = await costTracker.recordLLMCall({
      model: 'gemini-1.5-flash',
      tokensIn: 1200,
      tokensOut: 450,
      cachedTokensIn: 300
    });

    // Verify context accumulates tokens/cost in real time
    logger.info('Context metrics updated in real-time:', {
      totalTokensIn: activeCtx.totalTokensIn,
      totalTokensOut: activeCtx.totalTokensOut,
      totalCostUsd: activeCtx.totalCostUsd
    });

    // 7. Verify PgVectorStore persistence and RAG indexing
    logger.info('Initializing RAG database with PgVectorStore...');

    // Use a mock embedding generator to avoid network reliance in the demo if no key is set
    const geminiApiKey = process.env.GEMINI_API_KEY || 'dummy_api_key';
    const embeddings = new GeminiEmbeddings('gemini-embedding-001', { apiKey: geminiApiKey });
    
    // Override embedDocuments and embedQuery for offline demo stability
    embeddings.embedQuery = async () => Array.from({ length: 768 }, () => Math.random());
    embeddings.embedDocuments = async (texts) => texts.map(() => Array.from({ length: 768 }, () => Math.random()));

    const pgVectorStore = new PgVectorStore(embeddings, mockPg);
    await pgVectorStore.initializeSchema();

    const docItems: Document[] = [
      { pageContent: 'Microcity is the capital of Nanio Land, founded in 2026.', metadata: { category: 'history' } },
      { pageContent: 'Nanio is a lightweight typescript agentic framework featuring structured JSON observability.', metadata: { category: 'docs' } }
    ];

    const { elapsedMs } = await timeOperation(async () => {
      await pgVectorStore.addDocuments(docItems);
    });

    logger.info('Successfully indexed documents in PgVectorStore.');
    await perfTracker.record('embed_miss', elapsedMs);

    logger.info('Querying PgVectorStore...');
    const pgResults = await pgVectorStore.similaritySearch('Where is the capital of Nanio Land?');
    logger.info('Search results retrieved from PostgreSQL:', { results: pgResults });

    // 8. Verify QdrantVectorStore persistence
    logger.info('Initializing RAG database with QdrantVectorStore...');
    const qdrantStore = new QdrantVectorStore(embeddings, { url: 'http://localhost:6333' });
    await qdrantStore.initializeCollection();
    await qdrantStore.addDocuments(docItems);
    const qdrantResults = await qdrantStore.similaritySearch('typescript agentic framework');
    logger.info('Search results retrieved from Qdrant REST:', { results: qdrantResults });
    // 8b. Verify ZVecVectorStore persistence
    logger.info('Initializing RAG database with ZVecVectorStore...');
    const zvecPath = './zvec_data/test_vector_store';
    // Clean old collection directory if exists
    fs.rmSync(zvecPath, { recursive: true, force: true });
    const zvecStore = new ZVecVectorStore(embeddings, zvecPath);
    await zvecStore.addDocuments(docItems);
    const zvecResults = await zvecStore.similaritySearch('Where is the capital of Nanio Land?');
    logger.info('Search results retrieved from ZVecVectorStore:', { results: zvecResults });

    // -----------------------------------------------------------------------
    // 8c. ZVec + GeminiEmbeddings Chat Integration
    //     Shows ZVecVectorStore used as a chat knowledge-base with the Gemini
    //     cloud embedding provider. Uses the same 768-dim mock for demo stability.
    // -----------------------------------------------------------------------
    logger.info('=== 8c. ZVec + GeminiEmbeddings Chat Integration ===');
    const geminiChatZvecPath = './zvec_data/chat_gemini';
    fs.rmSync(geminiChatZvecPath, { recursive: true, force: true });
    // GeminiEmbeddings provider (768-dim) — mock embedQuery/embedDocuments for offline demo
    const geminiChatEmbeddings = new GeminiEmbeddings('gemini-embedding-001', { apiKey: 'demo' });
    geminiChatEmbeddings.embedQuery = async () => Array.from({ length: 768 }, () => Math.random());
    geminiChatEmbeddings.embedDocuments = async (texts) => texts.map(() => Array.from({ length: 768 }, () => Math.random()));

    const geminiZvecStore = new ZVecVectorStore(geminiChatEmbeddings, geminiChatZvecPath, 'chat_gemini');
    await geminiZvecStore.addDocuments([
      { pageContent: 'Nanio uses GeminiEmbeddings for cloud-scale semantic retrieval.', metadata: { source: 'docs', provider: 'gemini-embedding-001' } },
      { pageContent: 'ZVecVectorStore provides local HNSW indexing for millisecond retrieval.', metadata: { source: 'docs', provider: 'zvec' } },
      { pageContent: 'GeminiEmbeddings produces 768-dimensional vectors via batchEmbedContents API.', metadata: { source: 'api-ref', provider: 'gemini-embedding-001' } }
    ]);
    const geminiChatResults = await geminiZvecStore.similaritySearch('cloud embedding semantic search', 2);
    logger.info('ZVec + GeminiEmbeddings chat RAG results:', {
      provider: geminiChatEmbeddings.model,
      topK: 2,
      results: geminiChatResults.map(d => d.pageContent)
    });

    // -----------------------------------------------------------------------
    // 8d. ZVec + TransformersEmbeddings Chat Integration (local, no API key)
    //     Shows ZVecVectorStore used as a chat knowledge-base with the local
    //     SentenceTransformers provider. Uses 384-dim mock for demo stability.
    //     In production, replace the mock overrides with the real TransformersEmbeddings
    //     instance and it will download the ONNX model on first call.
    // -----------------------------------------------------------------------
    logger.info('=== 8d. ZVec + TransformersEmbeddings Chat Integration (local) ===');
    const localChatZvecPath = './zvec_data/chat_local';
    fs.rmSync(localChatZvecPath, { recursive: true, force: true });
    // TransformersEmbeddings provider (384-dim, Xenova/all-MiniLM-L6-v2)
    // Mock for demo: in production drop the overrides and use real ONNX inference
    const localChatEmbeddings = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');
    localChatEmbeddings.embedQuery = async () => Array.from({ length: 384 }, () => Math.random());
    localChatEmbeddings.embedDocuments = async (texts) => texts.map(() => Array.from({ length: 384 }, () => Math.random()));

    const localZvecStore = new ZVecVectorStore(localChatEmbeddings, localChatZvecPath, 'chat_local');
    await localZvecStore.addDocuments([
      { pageContent: 'TransformersEmbeddings runs entirely offline using Hugging Face ONNX Runtime.', metadata: { source: 'docs', provider: 'Xenova/all-MiniLM-L6-v2' } },
      { pageContent: 'SentenceTransformers all-MiniLM-L6-v2 produces 384-dimensional normalized vectors.', metadata: { source: 'model-card', provider: 'Xenova/all-MiniLM-L6-v2' } },
      { pageContent: 'Local embeddings require no API key and respect data privacy.', metadata: { source: 'docs', provider: 'Xenova/all-MiniLM-L6-v2' } }
    ]);
    const localChatResults = await localZvecStore.similaritySearch('local embedding no api key', 2);
    logger.info('ZVec + TransformersEmbeddings chat RAG results:', {
      provider: localChatEmbeddings.model,
      topK: 2,
      results: localChatResults.map(d => d.pageContent)
    });

    // Summary: ZVec accepts any BaseEmbeddings provider — Gemini (cloud, 768-dim),
    // OpenAI (cloud, 1536-dim), or TransformersEmbeddings (local, 384-dim).
    // Swap the provider in the ZVecVectorStore constructor with zero other changes.
    logger.info('Provider swap summary:', {
      geminiProvider: geminiChatEmbeddings.model,
      geminiDim: 768,
      localProvider: localChatEmbeddings.model,
      localDim: 384,
      openAIProvider: new OpenAIEmbeddings('text-embedding-3-small').model,
      openAIDim: 1536
    });
    // 9. Verify LLM Config Repository
    logger.info('Initializing PgLLMConfigRepository...');
    const configRepo = new PgLLMConfigRepository(mockPg);
    await configRepo.initializeSchema();
    const configFree = await configRepo.getForTier('FREE');
    logger.info('Retrieved FREE tier config parameters:', configFree);

    // 10. Verify PgChatMemory
    logger.info('Initializing PgChatMemory...');
    const chatMemory = new PgChatMemory(mockPg, activeCtx.sessionId, 'conversation_turns', activeCtx.userId);
    await chatMemory.initializeSchema();
    
    await chatMemory.addMessage({ role: 'user', content: 'What is the speed of gravity?' });
    await chatMemory.addMessage({ role: 'assistant', content: 'Gravity propagates at the speed of light.' });

    const messages = await chatMemory.getMessages();
    logger.info('Conversation history retrieved from PgChatMemory:', messages);

    // 11. Verify XAIModel and XAIImageClient structure and token counting
    logger.info('Verifying XAIModel and XAIImageClient...');
    const xaiModel = new XAIModel('grok-2', { apiKey: 'dummy_xai_key' });
    const xaiImgClient = new XAIImageClient('grok-imagine-image', { apiKey: 'dummy_xai_key' });
    
    const tokens = await xaiModel.countTokens('Hello xAI, grok-2 is nice.');
    logger.info(`Approximated tokens for text: ${tokens}`);

    // Verify image client token count is 0
    const imgTokens = await xaiImgClient.countTokens('draw a beautiful nanio agent');
    logger.info(`Approximated tokens for image client: ${imgTokens}`);

    // 12. Hash arguments demo
    const testArgs = { query: 'Microcity', limit: 5 };
    logger.info('Hashed operation arguments:', { raw: testArgs, hash: hashArgs(testArgs) });

    // 13. Verify LLMRegistry and ChatConfig dynamic routing
    logger.info('=== VERIFYING LLM REGISTRY & CHATCONFIG DYNAMIC ROUTING ===');

    class MockModel extends BaseModel {
      public name: string;
      public failWithRateLimit = false;
      public callCount = 0;

      constructor(name: string) {
        super();
        this.name = name;
      }

      async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
        this.callCount++;
        if (this.failWithRateLimit) {
          const err = new Error(`Rate limit hit on ${this.name}`);
          err.name = 'RateLimitError';
          throw err;
        }
        return {
          content: `Response from ${this.name}`,
          usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 }
        };
      }
    }

    const primaryModel = new MockModel('primary-mock');
    const lighterModel = new MockModel('lighter-mock');
    const freeModel = new MockModel('free-mock');

    const registry = new LLMRegistry(
      [
        { tier: 'PRIMARY', model: primaryModel, name: 'primary-mock' },
        { tier: 'LIGHTER', model: lighterModel, name: 'lighter-mock' },
        { tier: 'FREE', model: freeModel, name: 'free-mock' }
      ],
      costTracker
    );

    // Test A: Normal generation (starts at PRIMARY)
    const resNormal = await registry.generate(
      [{ role: 'user', content: 'hello' }],
      { config: new ChatConfig({ userId: 'test-user', modelTier: 'PRIMARY' }) }
    );
    logger.info(`Normal generation result: ${resNormal.content} (Primary calls: ${primaryModel.callCount}, Lighter calls: ${lighterModel.callCount})`);

    // Test B: RateLimit Failover (Primary fails, falls back to Lighter)
    primaryModel.failWithRateLimit = true;
    const resFailover = await registry.generate(
      [{ role: 'user', content: 'hello failover' }],
      { config: new ChatConfig({ userId: 'test-user', modelTier: 'PRIMARY' }) }
    );
    logger.info(`Failover generation result: ${resFailover.content} (Primary calls: ${primaryModel.callCount}, Lighter calls: ${lighterModel.callCount})`);

    // Test C: Budget Throttling (Simulate THROTTLED status, skips PRIMARY directly to LIGHTER)
    primaryModel.failWithRateLimit = false; // Reset
    primaryModel.callCount = 0;
    lighterModel.callCount = 0;

    // Record high costs in costTracker/pgCostStore for 'throttled-user' to force budget throttle status
    // FREE tier has limit of 0.20 USD. Let's record 0.195 USD to cross the 95% throttle threshold (0.19 USD)
    await mockPg.query('INSERT INTO llm_cost_events', [
      'chat', 'some-model', 'throttled-user', 'some-session', 'some-turn', 100000, 50000, '0.1950'
    ]);


    const resThrottled = await registry.generate(
      [{ role: 'user', content: 'hello budget' }],
      { config: new ChatConfig({ userId: 'throttled-user', userTier: 'FREE', modelTier: 'PRIMARY' }) }
    );
    logger.info(`Budget throttled generation result: ${resThrottled.content} (Primary calls: ${primaryModel.callCount}, Lighter calls: ${lighterModel.callCount})`);

    // 14. Verify IndexHydratedRAG (IHR) implementation
    logger.info('=== VERIFYING INDEX HYDRATED RAG (IHR) ===');
    
    class IhrMockModel extends BaseModel {
      public name = 'ihr-mock-model';
      public mockIngestResponse = JSON.stringify({
        '1': { related_ids: [], sibling_ids: [], prerequisite_ids: [] },
        '1.1': { related_ids: [], sibling_ids: ['1.2'], prerequisite_ids: ['1'] },
        '1.2': { related_ids: [], sibling_ids: ['1.1'], prerequisite_ids: ['1'] },
        '1.2.1': { related_ids: [], sibling_ids: [], prerequisite_ids: ['1.2'] }
      });
      public mockSearchResponse = JSON.stringify({
        sections: ['1.2.1']
      });
      public mockAnswerResponse = 'IndexHydratedRAG is fully operational with context tree lineage!';

      async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
        const hasSearch = messages.some(m => m.content && m.content.includes('You are a great tree search engineer'));
        const hasIngest = messages.some(m => m.content && m.content.includes('You are a document structure analyst'));
        const hasAnswer = messages.some(m => m.content && m.content.includes('answering queries based on the structured document context'));
        const hasStopwords = messages.some(m => m.content && m.content.includes('language-detection specialist'));

        if (hasIngest) {
          return {
            content: this.mockIngestResponse,
            usage: { promptTokens: 50, completionTokens: 50, totalTokens: 100 }
          };
        }

        if (hasStopwords) {
          return {
            content: JSON.stringify({
              language: 'English',
              languageCode: 'en',
              stopwords: ['the', 'a', 'an', 'and', 'or', 'but']
            }),
            usage: { promptTokens: 60, completionTokens: 25, totalTokens: 85 }
          };
        }

        if (hasSearch) {
          const hasToolResult = messages.some(m => m.role === 'tool');
          if (!hasToolResult) {
            const userMsg = messages.find(m => m.role === 'user')?.content || '';
            return {
              content: 'Calling search_sections tool...',
              toolCalls: [{
                id: 'call_search',
                name: 'search_sections',
                arguments: { query: userMsg }
              }]
            };
          }
          return {
            content: this.mockSearchResponse,
            usage: { promptTokens: 80, completionTokens: 20, totalTokens: 100 }
          };
        }

        if (hasAnswer) {
          return {
            content: this.mockAnswerResponse,
            usage: { promptTokens: 150, completionTokens: 30, totalTokens: 180 }
          };
        }

        return {
          content: 'Default mock response',
          usage: { promptTokens: 5, completionTokens: 5, totalTokens: 10 }
        };
      }
    }

    const ihrModel = new IhrMockModel();
    const ihrEmbeddings = new TransformersEmbeddings('Xenova/all-MiniLM-L6-v2');

    const ihr = new IndexHydratedRAG(mockPg, ihrEmbeddings, ihrModel);
    await ihr.initializeSchema();
    logger.info('IHR Database Schemas initialized.');

    const docId = 'doc_nanio_guidelines';
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
      },
      {
        section_id: '1.2',
        parent_id: '1',
        level: 2,
        heading: 'IndexHydratedRAG Design',
        content: 'IndexHydratedRAG uses a two-layer pointer indexing design to combine vector speed with SQL tree lineage.',
        summary: 'Details of the IndexHydratedRAG (IHR) retrieval design.'
      },
      {
        section_id: '1.2.1',
        parent_id: '1.2',
        level: 3,
        heading: 'Context Gating System',
        content: 'Context Gating prevents context token budget exhaustion by estimating tokens in real time.',
        summary: 'Detailed explanation of character token ratio budget gating.'
      }
    ];

    logger.info('Ingesting structured documents into IHR relational and vector tables...');
    await ihr.ingest(docId, 'Nanio Architecture Specification', 'https://nanio.dev/specs', sections);
    const { ZVecOpen } = await import('@zvec/zvec');
    const zcol = ZVecOpen(`./zvec_data/${docId}`);
    logger.info('Ingestion complete. Database table and Zvec stats:', {
      documents: mockPg.tables.documents.length,
      sections: mockPg.tables.section_table.length,
      zvecDocs: zcol.stats.docCount
    });
    zcol.closeSync();

    // Run Test 1: Fast Path (Small content)
    logger.info('Executing IHR query (Test 1: Fast Path)...');
    const responseFast = await ihr.retrieve('Explain the context gating architecture.', docId);
    logger.info('Fast path response retrieved:', {
      answer: responseFast.answer,
      path: responseFast.path,
      retrieved: responseFast.retrievedSections
    });
    logger.info('Context tree lineage for LLM:\n' + responseFast.context);

    // Run Test 2: TF-IDF Fallback (Large content exceeding gate)
    logger.info('Executing IHR query (Test 2: TF-IDF Fallback Path)...');
    const targetNode = mockPg.tables.section_table.find((n: any) => n.section_id === `${docId}__1.2.1`);
    if (targetNode) {
      targetNode.content = 'Context Gating prevents context token budget exhaustion. '.repeat(500); // 30,000+ chars
    }

    const responseTfidf = await ihr.retrieve('Explain the context gating architecture.', docId, { tfidfLimit: 1 });
    logger.info('TF-IDF fallback path response retrieved:', {
      answer: responseTfidf.answer,
      path: responseTfidf.path,
      retrieved: responseTfidf.retrievedSections
    });
    logger.info('Context tree lineage for LLM:\n' + responseTfidf.context);

    // Run Test 3: LLM Auto-Stopwords
    logger.info('Executing IHR query (Test 3: LLM Auto-Stopwords)...');
    const responseAutoStopwords = await ihr.retrieve('Explain the context gating architecture.', docId, {
      tfidfLimit: 1,
      autoStopwords: true
    });
    logger.info('LLM Auto-Stopwords path response retrieved:', {
      answer: responseAutoStopwords.answer,
      path: responseAutoStopwords.path,
      retrieved: responseAutoStopwords.retrievedSections
    });
    logger.info('Context tree lineage for LLM:\n' + responseAutoStopwords.context);

    // -----------------------------------------------------------------------
    // 15. Multi-Provider IHR — GeminiEmbeddings (768-dim cloud)
    //     Demonstrates IHR.ingest() and IHR.retrieve() with GeminiEmbeddings
    //     instead of TransformersEmbeddings. The embeddingProvider field in the
    //     structured logs will show 'gemini-embedding-001' instead of
    //     'Xenova/all-MiniLM-L6-v2', confirming pluggable provider swap.
    //     ZVecVectorStore dimension is auto-detected on first embedQuery call.
    // -----------------------------------------------------------------------
    logger.info('=== 15. IHR Multi-Provider Demo: GeminiEmbeddings (768-dim) ===');
    const geminiIhrEmbeddings = new GeminiEmbeddings('gemini-embedding-001', { apiKey: 'demo' });
    // Mock 768-dim for offline demo stability
    geminiIhrEmbeddings.embedQuery = async () => Array.from({ length: 768 }, () => Math.random());
    geminiIhrEmbeddings.embedDocuments = async (texts) => texts.map(() => Array.from({ length: 768 }, () => Math.random()));

    const geminiMockPg = new MockPgClient() as any;
    const ihrGemini = new IndexHydratedRAG(geminiMockPg, geminiIhrEmbeddings, ihrModel);
    await ihrGemini.initializeSchema();

    const docIdGemini = 'doc_gemini_provider_demo';
    await ihrGemini.ingest(docIdGemini, 'Gemini Provider Test Doc', 'https://nanio.dev/gemini-test', sections);
    logger.info('IHR ingestion with GeminiEmbeddings complete', {
      provider: ihrGemini.embeddingProviderName,
      dim: 768,
      docId: docIdGemini
    });

    const geminiIhrResponse = await ihrGemini.retrieve('Explain the context gating architecture.', docIdGemini);
    logger.info('IHR + GeminiEmbeddings retrieval result:', {
      provider: ihrGemini.embeddingProviderName,
      path: geminiIhrResponse.path,
      retrieved: geminiIhrResponse.retrievedSections,
      answerSnippet: geminiIhrResponse.answer.slice(0, 80)
    });

    // Verify provider swap is logged — TransformersEmbeddings IHR (already tested in section 14)
    logger.info('IHR provider comparison:', {
      section14Provider: ihr.embeddingProviderName,  // Xenova/all-MiniLM-L6-v2
      section15Provider: ihrGemini.embeddingProviderName  // gemini-embedding-001
    });
  });

  logger.info('=== DEMO EXECUTION COMPLETE ===');
}

runDemo();
