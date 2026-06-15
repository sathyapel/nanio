import * as dotenv from 'dotenv';
import { z } from 'zod';
import { 
  SimpleAgent, 
  ToolCallingAgent, 
  BaseModel, 
  Message, 
  LLMResponse, 
  GenerateOptions,
  PgLLMConfigRepository,
  PgChatMemory
} from '@nanio/core';
import { FunctionTool } from '@nanio/tools';
import { GeminiModel, BaseAIClient, ClaudeModel, XAIModel, XAIImageClient } from '@nanio/providers';
import { GeminiEmbeddings } from '@nanio/embeddings';
import { MemoryVectorStore, MongoVectorStore, PgVectorStore, QdrantVectorStore, Document } from '@nanio/vectorstore';
import { FallbackRouter } from '@nanio/registry';
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
    conversation_turns: []
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
  });

  logger.info('=== DEMO EXECUTION COMPLETE ===');
}

runDemo();
