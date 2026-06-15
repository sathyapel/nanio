import { createHash } from 'crypto';
import { getContext } from './context.js';

export class LoggerNotConfigured extends Error {
  constructor(message?: string) {
    super(message || 'Logger was not initialized. Call setupLogging() at startup.');
    this.name = 'LoggerNotConfigured';
  }
}

/**
 * Event name constants. All logs use structured event keys to allow easy search and filter.
 */
export class Events {
  // LLM completions & chats
  static readonly LLM_CHAT_START = 'llm.chat.start';
  static readonly LLM_CHAT_COMPLETE = 'llm.chat.complete';
  static readonly LLM_CHAT_STREAM_FIRST_TOKEN = 'llm.chat.stream.first_token';
  static readonly LLM_CONTEXT_TRIMMED = 'llm.context.trimmed';
  static readonly LLM_DEADLINE_EXCEEDED = 'llm.deadline.exceeded';

  // Base client resiliency
  static readonly CLIENT_RETRY = 'client.retry';
  static readonly CLIENT_CIRCUIT_OPEN = 'client.circuit.open';
  static readonly CLIENT_CIRCUIT_HALF_OPEN = 'client.circuit.half_open';
  static readonly CLIENT_CIRCUIT_CLOSED = 'client.circuit.closed';
  static readonly CLIENT_RATE_LIMIT_WAIT = 'client.rate_limit.wait';
  static readonly CLIENT_INITIALIZED = 'client.initialized';
  static readonly CLIENT_CLOSED = 'client.closed';

  // Model registries
  static readonly REGISTRY_FALLBACK = 'registry.model.fallback';
  static readonly REGISTRY_ALL_FAILED = 'registry.all_providers.failed';
  static readonly REGISTRY_MODEL_SELECTED = 'registry.model.selected';

  // Embeddings
  static readonly EMBED_COMPLETE = 'embed.request.complete';
  static readonly EMBED_BATCH_COMPLETE = 'embed.batch.complete';
  static readonly EMBED_CACHE_HIT = 'embed.cache.hit';

  // Tools lifecycle
  static readonly TOOL_START = 'tool.call.start';
  static readonly TOOL_COMPLETE = 'tool.call.complete';
  static readonly TOOL_TIMEOUT = 'tool.call.timeout';
  static readonly TOOL_ERROR = 'tool.call.error';

  // Cost tracking
  static readonly COST_BUDGET_WARNING = 'cost.budget.warning';
  static readonly COST_BUDGET_EXCEEDED = 'cost.budget.exceeded';
  static readonly COST_BUDGET_THROTTLED = 'cost.budget.throttled';
  static readonly COST_GLOBAL_WARNING = 'cost.global.warning';
  static readonly COST_GLOBAL_HARD_STOP = 'cost.global.hard_stop';
  static readonly COST_TRACKER_DEGRADED = 'cost.tracker.degraded';

  // Database
  static readonly DB_QUERY_EXECUTED = 'db.query.executed';
  static readonly DB_QUERY_SLOW = 'db.query.slow';
  static readonly DB_QUERY_ERROR = 'db.query.error';

  // Memory & Vector Search
  static readonly MEMORY_SEARCH = 'memory.search.complete';
  static readonly MEMORY_SEARCH_SLOW = 'memory.search.slow';
  static readonly MEMORY_BATCH_WRITTEN = 'memory.batch.written';

  // General Routine
  static readonly ROUTINE_ERROR = 'routine.unhandled_error';
}

// Global logger configurations
let globalLogLevel = 'INFO';
let isConfigured = false;

export function setupLogging(level: string = 'INFO'): void {
  const envLevel = process.env.LOG_LEVEL || level;
  globalLogLevel = envLevel.toUpperCase();

  // Enforce no DEBUG in production
  if (process.env.NODE_ENV === 'production' && globalLogLevel === 'DEBUG') {
    globalLogLevel = 'INFO';
  }

  isConfigured = true;
}

// Log level weighting
const LOG_LEVELS: Record<string, number> = {
  DEBUG: 10,
  INFO: 20,
  WARN: 30,
  ERROR: 40,
  CRITICAL: 50
};

// PII Regex patterns for phone, email, and API key scrubbing
const PII_PATTERNS = [
  { pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, replacement: '[email]' },
  { pattern: /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g, replacement: '[phone]' },
  { pattern: /\b[A-Za-z0-9_-]{32,}\b/g, replacement: '[secret]' }
];

const ALWAYS_SCRUB_FIELDS = new Set(['password', 'token', 'api_key', 'apikey', 'secret', 'authorization']);
const INFO_SCRUB_FIELDS = new Set(['prompt', 'text', 'content', 'message', 'transcript', 'response']);

function scrubString(val: string): string {
  let scrubbed = val;
  for (const { pattern, replacement } of PII_PATTERNS) {
    scrubbed = scrubbed.replace(pattern, replacement);
  }
  return scrubbed;
}

function scrubValue(key: string, val: any, logLevel: string): any {
  const keyLower = key.toLowerCase();

  if (ALWAYS_SCRUB_FIELDS.has(keyLower)) {
    return '[redacted]';
  }

  const currentWeight = LOG_LEVELS[logLevel] || 20;
  if (currentWeight >= LOG_LEVELS.INFO && INFO_SCRUB_FIELDS.has(keyLower)) {
    return '[scrubbed-at-info]';
  }

  if (typeof val === 'string') {
    return scrubString(val);
  }

  if (val && typeof val === 'object') {
    if (Array.isArray(val)) {
      return val.map(item => scrubValue(key, item, logLevel));
    }
    const scrubbedObj: Record<string, any> = {};
    for (const [k, v] of Object.entries(val)) {
      scrubbedObj[k] = scrubValue(k, v, logLevel);
    }
    return scrubbedObj;
  }

  return val;
}

function scrubRecord(record: Record<string, any>, logLevel: string): Record<string, any> {
  const scrubbed: Record<string, any> = {};
  for (const [k, v] of Object.entries(record)) {
    scrubbed[k] = scrubValue(k, v, logLevel);
  }
  return scrubbed;
}

function scrubUserId(userId: string): string {
  if (!userId || userId.length < 4) {
    return '***';
  }
  return `***${userId.slice(-4)}`;
}

/**
 * Hash arguments for safe identification of unique payloads.
 */
export function hashArgs(args: any): string {
  try {
    const serialized = JSON.stringify(args, Object.keys(args || {}).sort());
    return createHash('sha256').update(serialized).digest('hex').substring(0, 8);
  } catch {
    return createHash('sha256').update(String(args)).digest('hex').substring(0, 8);
  }
}

/**
 * Custom structured context logger wrapper.
 */
export class JSONLogger {
  constructor(private moduleName: string) {}

  private log(level: string, event: string, extra?: Record<string, any>): void {
    const targetLevelWeight = LOG_LEVELS[level] || 20;
    const currentLevelWeight = LOG_LEVELS[globalLogLevel] || 20;

    if (targetLevelWeight < currentLevelWeight) {
      return;
    }

    const ctx = getContext(false);

    let logObj: Record<string, any> = {
      ts: new Date().toISOString(),
      level,
      event,
      module: this.moduleName,
      trace_id: ctx.traceId,
      turn_id: ctx.turnId,
      session_id: ctx.sessionId,
      user_id: scrubUserId(ctx.userId),
      stage: ctx.stage,
      model_used: ctx.modelUsed
    };

    if (extra) {
      logObj = { ...logObj, ...extra };
    }

    const finalRecord = scrubRecord(logObj, level);
    console.log(JSON.stringify(finalRecord));
  }

  debug(event: string, extra?: Record<string, any>): void {
    this.log('DEBUG', event, extra);
  }

  info(event: string, extra?: Record<string, any>): void {
    this.log('INFO', event, extra);
  }

  warn(event: string, extra?: Record<string, any>): void {
    this.log('WARN', event, extra);
  }

  error(event: string, extra?: Record<string, any>, err?: Error): void {
    const errorFields: Record<string, any> = {};
    if (err) {
      errorFields.error_message = err.message;
      errorFields.error_stack = err.stack;
    }
    this.log('ERROR', event, { ...extra, ...errorFields });
  }

  critical(event: string, extra?: Record<string, any>, err?: Error): void {
    const errorFields: Record<string, any> = {};
    if (err) {
      errorFields.error_message = err.message;
      errorFields.error_stack = err.stack;
    }
    this.log('CRITICAL', event, { ...extra, ...errorFields });
  }
}

/**
 * Logger factory method.
 */
export function getLogger(name: string): JSONLogger {
  if (!isConfigured) {
    console.warn(
      `[observability] WARNING: getLogger('${name}') called before setupLogging(). Call setupLogging() at application startup.`
    );
  }
  return new JSONLogger(name);
}
