import { getLogger } from './logger.js';
import pg from 'pg';

const logger = getLogger('PerfTracker');

export interface LatencyBudget {
  p50Ms: number;
  p95Ms: number;
  hardLimitMs: number;
}

export const BUDGETS: Record<string, LatencyBudget> = {
  first_token: { p50Ms: 800, p95Ms: 2000, hardLimitMs: 5000 },
  full_turn_no_tools: { p50Ms: 2000, p95Ms: 5000, hardLimitMs: 15000 },
  full_turn_with_tools: { p50Ms: 3500, p95Ms: 8000, hardLimitMs: 20000 },
  tool_call: { p50Ms: 500, p95Ms: 5000, hardLimitMs: 8000 },
  embed_miss: { p50Ms: 100, p95Ms: 300, hardLimitMs: 1000 },
  embed_hit: { p50Ms: 1, p95Ms: 5, hardLimitMs: 50 },
  image_gen: { p50Ms: 8000, p95Ms: 15000, hardLimitMs: 30000 },
  live_first_audio_chunk: { p50Ms: 400, p95Ms: 800, hardLimitMs: 1500 },
  count_tokens: { p50Ms: 50, p95Ms: 200, hardLimitMs: 500 }
};

const HISTOGRAM_MAX_SAMPLES = 500;

/**
 * Base abstract performance storage store.
 */
export abstract class BasePerfStore {
  abstract recordLatency(metric: string, latencyMs: number, maxSamples?: number): Promise<void>;
  abstract getPercentile(metric: string, pct: number): Promise<number | null>;
  abstract incrementCounter(key: string, amount?: number): Promise<void>;
  abstract getCounter(key: string): Promise<number>;
  abstract setGauge(key: string, value: number, ttlSeconds?: number): Promise<void>;
}

/**
 * Zero-dependency in-memory performance store.
 */
export class MemoryPerfStore extends BasePerfStore {
  private histograms = new Map<string, number[]>();
  private counters = new Map<string, number>();
  private gauges = new Map<string, number>();

  async recordLatency(metric: string, latencyMs: number, maxSamples: number = HISTOGRAM_MAX_SAMPLES): Promise<void> {
    const list = this.histograms.get(metric) || [];
    list.push(latencyMs);
    if (list.length > maxSamples) {
      list.shift(); // Evict oldest
    }
    this.histograms.set(metric, list);
  }

  async getPercentile(metric: string, pct: number): Promise<number | null> {
    const list = this.histograms.get(metric);
    if (!list || list.length === 0) return null;
    const sorted = [...list].sort((a, b) => a - b);
    const index = Math.min(Math.floor(sorted.length * pct), sorted.length - 1);
    return sorted[index];
  }

  async incrementCounter(key: string, amount: number = 1): Promise<void> {
    const count = this.counters.get(key) || 0;
    this.counters.set(key, count + amount);
  }

  async getCounter(key: string): Promise<number> {
    return this.counters.get(key) || 0;
  }

  async setGauge(key: string, value: number): Promise<void> {
    this.gauges.set(key, value);
  }
}

/**
 * Redis performance store using atomic ZADD metrics pruning pipeline commands.
 */
export class RedisPerfStore extends BasePerfStore {
  constructor(private redis: any) {
    super();
  }

  async recordLatency(metric: string, latencyMs: number, maxSamples: number = HISTOGRAM_MAX_SAMPLES): Promise<void> {
    const key = `perf:${metric}`;
    const now = Date.now() / 1000;
    const member = `${latencyMs}:${now}`;

    const pipeline = this.redis.pipeline();
    pipeline.zadd(key, now, member);
    pipeline.zremrangebyrank(key, 0, -(maxSamples + 1));
    pipeline.expire(key, 3600);

    await pipeline.exec();
  }

  async getPercentile(metric: string, pct: number): Promise<number | null> {
    const key = `perf:${metric}`;
    const members: string[] = await this.redis.zrange(key, 0, -1);
    if (!members || members.length === 0) return null;

    const values = members
      .map(m => parseFloat(m.split(':')[0]))
      .sort((a, b) => a - b);

    const index = Math.min(Math.floor(values.length * pct), values.length - 1);
    return values[index];
  }

  async incrementCounter(key: string, amount: number = 1): Promise<void> {
    const redisKey = `perf:counter:${key}`;
    const pipeline = this.redis.pipeline();
    pipeline.incrby(redisKey, amount);
    pipeline.expire(redisKey, 3600);
    await pipeline.exec();
  }

  async getCounter(key: string): Promise<number> {
    const redisKey = `perf:counter:${key}`;
    const val = await this.redis.get(redisKey);
    return parseInt(val || '0');
  }

  async setGauge(key: string, value: number, ttlSeconds: number = 30): Promise<void> {
    const redisKey = `perf:gauge:${key}`;
    await this.redis.setex(redisKey, ttlSeconds, String(value));
  }
}

/**
 * Tracks and alerts on latency metrics.
 */
export class PerfTracker {
  constructor(private store: BasePerfStore = new MemoryPerfStore()) {}

  async record(metric: string, latencyMs: number): Promise<void> {
    const budget = BUDGETS[metric];

    if (budget && latencyMs > budget.hardLimitMs) {
      logger.warn('perf.hard_limit_exceeded', {
        metric,
        latency_ms: latencyMs,
        hard_limit_ms: budget.hardLimitMs
      });
    }

    try {
      await this.store.recordLatency(metric, latencyMs);
    } catch (err: any) {
      logger.warn('perf.tracker.degraded', { metric, error: err.message || err });
    }
  }

  async recordFirstToken(latencyMs: number): Promise<void> {
    await this.record('first_token', latencyMs);
  }

  async recordTurnLatency(latencyMs: number, hasTools: boolean): Promise<void> {
    const metric = hasTools ? 'full_turn_with_tools' : 'full_turn_no_tools';
    await this.record(metric, latencyMs);
  }

  async recordToolCall(toolName: string, latencyMs: number): Promise<void> {
    await this.record('tool_call', latencyMs);
    try {
      await this.store.recordLatency(`tool:${toolName}`, latencyMs);
    } catch (err: any) {
      logger.warn('perf.tracker.degraded', { metric: `tool:${toolName}`, error: err.message || err });
    }
  }

  async incrementRetry(clientName: string): Promise<void> {
    try {
      await this.store.incrementCounter(`retry:${clientName}`);
    } catch (err: any) {
      logger.warn('perf.counter.degraded', { counter: `retry:${clientName}`, error: err.message || err });
    }
  }

  async incrementCircuitOpen(clientName: string): Promise<void> {
    try {
      await this.store.incrementCounter(`circuit:${clientName}`);
    } catch (err: any) {
      logger.warn('perf.counter.degraded', { counter: `circuit:${clientName}`, error: err.message || err });
    }
  }

  async getPercentile(metric: string, pct: number): Promise<number | null> {
    try {
      return await this.store.getPercentile(metric, pct);
    } catch {
      return null;
    }
  }
}

/**
 * PostgreSQL-backed performance store.
 */
export class PgPerfStore extends BasePerfStore {
  constructor(
    private client: pg.Client | pg.Pool,
    private latenciesTable: string = 'perf_latencies',
    private countersTable: string = 'perf_counters',
    private gaugesTable: string = 'perf_gauges'
  ) {
    super();
  }

  async initializeSchema(): Promise<void> {
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS ${this.latenciesTable} (
        id SERIAL PRIMARY KEY,
        metric TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
      CREATE TABLE IF NOT EXISTS ${this.countersTable} (
        key TEXT PRIMARY KEY,
        value INT NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS ${this.gaugesTable} (
        key TEXT PRIMARY KEY,
        value REAL NOT NULL,
        expires_at TIMESTAMPTZ
      );
      CREATE INDEX IF NOT EXISTS idx_perf_latencies_metric ON ${this.latenciesTable} (metric, timestamp);
    `);
  }

  async recordLatency(metric: string, latencyMs: number, maxSamples: number = HISTOGRAM_MAX_SAMPLES): Promise<void> {
    await this.client.query(
      `INSERT INTO ${this.latenciesTable} (metric, latency_ms) VALUES ($1, $2)`,
      [metric, latencyMs]
    );
    // Delete excess samples
    await this.client.query(
      `DELETE FROM ${this.latenciesTable} WHERE id IN (
        SELECT id FROM ${this.latenciesTable} WHERE metric = $1 ORDER BY timestamp DESC OFFSET $2
      )`,
      [metric, maxSamples]
    );
  }

  async getPercentile(metric: string, pct: number): Promise<number | null> {
    const res = await this.client.query(
      `SELECT latency_ms FROM ${this.latenciesTable} WHERE metric = $1 ORDER BY latency_ms ASC`,
      [metric]
    );
    if (res.rows.length === 0) return null;
    const values = res.rows.map(row => parseFloat(row.latency_ms));
    const index = Math.min(Math.floor(values.length * pct), values.length - 1);
    return values[index];
  }

  async incrementCounter(key: string, amount: number = 1): Promise<void> {
    await this.client.query(
      `INSERT INTO ${this.countersTable} (key, value) VALUES ($1, $2)
       ON CONFLICT (key) DO UPDATE SET value = ${this.countersTable}.value + EXCLUDED.value`,
      [key, amount]
    );
  }

  async getCounter(key: string): Promise<number> {
    const res = await this.client.query(
      `SELECT value FROM ${this.countersTable} WHERE key = $1`,
      [key]
    );
    if (res.rows.length === 0) return 0;
    return res.rows[0].value;
  }

  async setGauge(key: string, value: number, ttlSeconds?: number): Promise<void> {
    const expiresAt = ttlSeconds ? new Date(Date.now() + ttlSeconds * 1000) : null;
    await this.client.query(
      `INSERT INTO ${this.gaugesTable} (key, value, expires_at) VALUES ($1, $2, $3)
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, expires_at = EXCLUDED.expires_at`,
      [key, value, expiresAt]
    );
  }
}
