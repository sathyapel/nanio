import { getContext } from './context.js';
import { getLogger, Events } from './logger.js';
import pg from 'pg';

const logger = getLogger('CostTracker');

export enum BudgetStatus {
  OK = 'OK',
  WARNING = 'WARNING',
  THROTTLED = 'THROTTLED',
  EXCEEDED = 'EXCEEDED',
  GLOBAL_WARNING = 'GLOBAL_WARNING',
  GLOBAL_HARD_STOP = 'GLOBAL_HARD_STOP'
}

export interface BudgetConfig {
  dailyLimitUsd: number;
  monthlyLimitUsd: number;
  alertThresholdPct: number;
  throttleThresholdPct: number;
}

export interface DailyCostSummary {
  totalUsd: number;
  llmUsd: number;
  liveUsd: number;
  imageUsd: number;
  imageEditUsd: number;
  embedUsd: number;
  turnCount: number;
  date: string;
}

export interface ModelPricing {
  inputPer1mTokens: number;
  outputPer1mTokens: number;
  cachedInputPer1mTokens?: number;
}

// Pricing configs
export const LLM_PRICING: Record<string, ModelPricing> = {
  'gemini-2.0-flash': { inputPer1mTokens: 0.075, outputPer1mTokens: 0.30, cachedInputPer1mTokens: 0.01875 },
  'gemini-2.0-flash-lite': { inputPer1mTokens: 0.0375, outputPer1mTokens: 0.15, cachedInputPer1mTokens: 0.01 },
  'gemini-1.5-flash': { inputPer1mTokens: 0.075, outputPer1mTokens: 0.30, cachedInputPer1mTokens: 0.01875 },
  'gpt-4o-mini': { inputPer1mTokens: 0.15, outputPer1mTokens: 0.60 }
};

export const EMBED_COST_PER_1K_CHARS: Record<string, number> = {
  'embedding-001': 0.000025,
  'text-embedding-004': 0.000025
};

export const TIER_BUDGETS: Record<string, BudgetConfig> = {
  FREE: { dailyLimitUsd: 0.20, monthlyLimitUsd: 0.50, alertThresholdPct: 0.80, throttleThresholdPct: 0.95 },
  LIGHTER: { dailyLimitUsd: 0.50, monthlyLimitUsd: 5.00, alertThresholdPct: 0.80, throttleThresholdPct: 0.95 },
  PRIMARY: { dailyLimitUsd: 2.00, monthlyLimitUsd: 20.00, alertThresholdPct: 0.80, throttleThresholdPct: 0.95 },
  HEAVY: { dailyLimitUsd: 5.00, monthlyLimitUsd: 100.00, alertThresholdPct: 0.90, throttleThresholdPct: 0.95 }
};

export const PLATFORM_DAILY_LIMIT_USD = 500.0;

function today(): string {
  return new Date().toISOString().split('T')[0];
}

function thisMonth(): string {
  return new Date().toISOString().substring(0, 7); // YYYY-MM
}

/**
 * Base abstract class for cost persistence stores.
 */
export abstract class BaseCostStore {
  abstract checkBudget(
    userId: string,
    tier: string,
    limits: BudgetConfig
  ): Promise<BudgetStatus>;

  abstract incrementCost(params: {
    userId: string;
    turnId: string;
    costUsd: number;
    operation: string;
    service: string;
    model: string;
    inputTokens?: number;
    outputTokens?: number;
    sessionId?: string;
  }): Promise<void>;

  abstract getDailyCost(userId: string, date: string): Promise<DailyCostSummary>;
}

/**
 * In-memory fallback cost store.
 */
export class MemoryCostStore extends BaseCostStore {
  private dailyCosts = new Map<string, Record<string, number>>(); // key: userId:date, fields: spent, turnCount, etc.
  private monthlyCosts = new Map<string, number>(); // key: userId:month
  private globalDailyCost = new Map<string, number>(); // key: date

  async checkBudget(userId: string, tier: string, limits: BudgetConfig): Promise<BudgetStatus> {
    const t = today();
    const m = thisMonth();

    const dailyKey = `${userId}:${t}`;
    const monthlyKey = `${userId}:${m}`;

    const dailySpent = this.dailyCosts.get(dailyKey)?.total || 0;
    const monthlySpent = this.monthlyCosts.get(monthlyKey) || 0;
    const globalSpent = this.globalDailyCost.get(t) || 0;

    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD) {
      return BudgetStatus.GLOBAL_HARD_STOP;
    }
    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD * 0.80) {
      return BudgetStatus.GLOBAL_WARNING;
    }

    if (dailySpent >= limits.dailyLimitUsd || monthlySpent >= limits.monthlyLimitUsd) {
      return BudgetStatus.EXCEEDED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.throttleThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.throttleThresholdPct
    ) {
      return BudgetStatus.THROTTLED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.alertThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.alertThresholdPct
    ) {
      return BudgetStatus.WARNING;
    }

    return BudgetStatus.OK;
  }

  async incrementCost(params: {
    userId: string;
    costUsd: number;
    operation: string;
  }): Promise<void> {
    const t = today();
    const m = thisMonth();

    const dailyKey = `${params.userId}:${t}`;
    const monthlyKey = `${params.userId}:${m}`;

    // Update daily
    const daily = this.dailyCosts.get(dailyKey) || { total: 0, llm: 0, embed: 0, live: 0, turnCount: 0 };
    daily.total += params.costUsd;
    if (params.operation === 'chat_or_generate_usd') {
      daily.llm += params.costUsd;
      daily.turnCount += 1;
    } else if (params.operation === 'embed_usd') {
      daily.embed += params.costUsd;
    } else if (params.operation === 'live_usd') {
      daily.live += params.costUsd;
    }
    this.dailyCosts.set(dailyKey, daily);

    // Update monthly
    const monthly = this.monthlyCosts.get(monthlyKey) || 0;
    this.monthlyCosts.set(monthlyKey, monthly + params.costUsd);

    // Update global
    const globalVal = this.globalDailyCost.get(t) || 0;
    this.globalDailyCost.set(t, globalVal + params.costUsd);
  }

  async getDailyCost(userId: string, date: string): Promise<DailyCostSummary> {
    const daily = this.dailyCosts.get(`${userId}:${date}`) || { total: 0, llm: 0, embed: 0, live: 0, turnCount: 0 };
    return {
      totalUsd: daily.total,
      llmUsd: daily.llm,
      liveUsd: daily.live,
      imageUsd: 0,
      imageEditUsd: 0,
      embedUsd: daily.embed,
      turnCount: daily.turnCount,
      date
    };
  }
}

/**
 * High-performance Redis cost store.
 */
export class RedisCostStore extends BaseCostStore {
  constructor(private redis: any) {
    super();
  }

  async checkBudget(userId: string, tier: string, limits: BudgetConfig): Promise<BudgetStatus> {
    const t = today();
    const m = thisMonth();

    const dailyKey = `cost:user:${userId}:daily:${t}`;
    const monthlyKey = `cost:user:${userId}:monthly:${m}`;
    const globalKey = `cost:global:daily:${t}`;

    const [dailySpentRaw, monthlySpentRaw, globalSpentRaw] = await Promise.all([
      this.redis.hget(dailyKey, 'total_usd'),
      this.redis.hget(monthlyKey, 'total_usd'),
      this.redis.hget(globalKey, 'total_usd')
    ]);

    const dailySpent = parseFloat(dailySpentRaw || '0');
    const monthlySpent = parseFloat(monthlySpentRaw || '0');
    const globalSpent = parseFloat(globalSpentRaw || '0');

    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD) {
      return BudgetStatus.GLOBAL_HARD_STOP;
    }
    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD * 0.8) {
      return BudgetStatus.GLOBAL_WARNING;
    }

    if (dailySpent >= limits.dailyLimitUsd || monthlySpent >= limits.monthlyLimitUsd) {
      return BudgetStatus.EXCEEDED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.throttleThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.throttleThresholdPct
    ) {
      return BudgetStatus.THROTTLED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.alertThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.alertThresholdPct
    ) {
      return BudgetStatus.WARNING;
    }

    return BudgetStatus.OK;
  }

  async incrementCost(params: {
    userId: string;
    turnId: string;
    costUsd: number;
    operation: string;
    service: string;
    model: string;
    inputTokens?: number;
    outputTokens?: number;
    sessionId?: string;
  }): Promise<void> {
    const t = today();
    const m = thisMonth();

    const turnKey = `cost:user:${params.userId}:turn:${params.turnId}`;
    const dailyKey = `cost:user:${params.userId}:daily:${t}`;
    const monthlyKey = `cost:user:${params.userId}:monthly:${m}`;
    const globalKey = `cost:global:daily:${t}`;

    const payload = JSON.stringify({
      timestamp: new Date().toISOString(),
      user_id: params.userId,
      turn_id: params.turnId,
      service: params.service,
      model: params.model,
      cost_usd: params.costUsd
    });

    const pipeline = this.redis.pipeline();

    pipeline.hincrbyfloat(turnKey, 'total_usd', params.costUsd);
    pipeline.hincrbyfloat(turnKey, params.operation, params.costUsd);
    pipeline.expire(turnKey, 86400);

    pipeline.hincrbyfloat(dailyKey, 'total_usd', params.costUsd);
    pipeline.hincrbyfloat(dailyKey, params.operation, params.costUsd);
    if (params.operation === 'chat_or_generate_usd') {
      pipeline.hincrby(dailyKey, 'turn_count', 1);
    }
    pipeline.expire(dailyKey, 32 * 86400);

    pipeline.hincrbyfloat(monthlyKey, 'total_usd', params.costUsd);
    pipeline.expire(monthlyKey, 400 * 86400);

    pipeline.hincrbyfloat(globalKey, 'total_usd', params.costUsd);
    pipeline.expire(globalKey, 400 * 86400);

    pipeline.rpush('cost:events:queue', payload);

    await pipeline.exec();
  }

  async getDailyCost(userId: string, date: string): Promise<DailyCostSummary> {
    const key = `cost:user:${userId}:daily:${date}`;
    const data = await this.redis.hgetall(key);
    return {
      totalUsd: parseFloat(data.total_usd || '0'),
      llmUsd: parseFloat(data.chat_or_generate_usd || '0'),
      liveUsd: parseFloat(data.live_usd || '0'),
      imageUsd: 0,
      imageEditUsd: 0,
      embedUsd: parseFloat(data.embed_usd || '0'),
      turnCount: parseInt(data.turn_count || '0'),
      date
    };
  }
}

/**
 * MongoDB cost store for persistent cost audit logging and budget settings.
 */
export class MongoCostStore extends BaseCostStore {
  constructor(private db: any) {
    super();
  }

  async checkBudget(userId: string, tier: string, limits: BudgetConfig): Promise<BudgetStatus> {
    const t = today();
    const m = thisMonth();

    // Query daily cost summaries
    const [userDaily, userMonthly, globalDaily] = await Promise.all([
      this.db.collection('user_daily_costs').findOne({ userId, date: t }),
      this.db.collection('user_monthly_costs').findOne({ userId, month: m }),
      this.db.collection('global_daily_costs').findOne({ date: t })
    ]);

    const dailySpent = userDaily?.totalUsd || 0;
    const monthlySpent = userMonthly?.totalUsd || 0;
    const globalSpent = globalDaily?.totalUsd || 0;

    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD) {
      return BudgetStatus.GLOBAL_HARD_STOP;
    }
    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD * 0.8) {
      return BudgetStatus.GLOBAL_WARNING;
    }

    if (dailySpent >= limits.dailyLimitUsd || monthlySpent >= limits.monthlyLimitUsd) {
      return BudgetStatus.EXCEEDED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.throttleThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.throttleThresholdPct
    ) {
      return BudgetStatus.THROTTLED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.alertThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.alertThresholdPct
    ) {
      return BudgetStatus.WARNING;
    }

    return BudgetStatus.OK;
  }

  async incrementCost(params: {
    userId: string;
    turnId: string;
    costUsd: number;
    operation: string;
    service: string;
    model: string;
    inputTokens?: number;
    outputTokens?: number;
    sessionId?: string;
  }): Promise<void> {
    const t = today();
    const m = thisMonth();

    // 1. Insert transaction detail log
    const eventDoc = {
      timestamp: new Date(),
      userId: params.userId,
      turnId: params.turnId,
      service: params.service,
      model: params.model,
      costUsd: params.costUsd,
      inputTokens: params.inputTokens || 0,
      outputTokens: params.outputTokens || 0,
      sessionId: params.sessionId
    };

    const isLlm = params.operation === 'chat_or_generate_usd';

    // 2. Increment summaries atomically
    await Promise.all([
      this.db.collection('cost_events').insertOne(eventDoc),

      this.db.collection('user_daily_costs').updateOne(
        { userId: params.userId, date: t },
        {
          $inc: {
            totalUsd: params.costUsd,
            [params.operation]: params.costUsd,
            ...(isLlm ? { turnCount: 1 } : {})
          }
        },
        { upsert: true }
      ),

      this.db.collection('user_monthly_costs').updateOne(
        { userId: params.userId, month: m },
        { $inc: { totalUsd: params.costUsd } },
        { upsert: true }
      ),

      this.db.collection('global_daily_costs').updateOne(
        { date: t },
        { $inc: { totalUsd: params.costUsd } },
        { upsert: true }
      )
    ]);
  }

  async getDailyCost(userId: string, date: string): Promise<DailyCostSummary> {
    const userDaily = await this.db.collection('user_daily_costs').findOne({ userId, date });
    return {
      totalUsd: userDaily?.totalUsd || 0,
      llmUsd: userDaily?.chat_or_generate_usd || 0,
      liveUsd: userDaily?.live_usd || 0,
      imageUsd: 0,
      imageEditUsd: 0,
      embedUsd: userDaily?.embed_usd || 0,
      turnCount: userDaily?.turnCount || 0,
      date
    };
  }
}

/**
 * Coordinates budget checking and dynamic cost tracking.
 */
export class CostTracker {
  constructor(private store: BaseCostStore = new MemoryCostStore()) {}

  async checkBudget(userId: string, tier: string = 'FREE'): Promise<BudgetStatus> {
    try {
      const config = TIER_BUDGETS[tier] || TIER_BUDGETS.FREE;
      return await this.store.checkBudget(userId, tier, config);
    } catch (err: any) {
      logger.warn(Events.COST_TRACKER_DEGRADED, { error: err.message || err });
      return BudgetStatus.OK; // Fail open
    }
  }

  private calcLlmCost(model: string, tokensIn: number, tokensOut: number, cachedTokensIn: number): number {
    const pricing = LLM_PRICING[model] || { inputPer1mTokens: 3.0, outputPer1mTokens: 15.0 };
    const effectiveInput = Math.max(0, tokensIn - cachedTokensIn);
    const cachedCost = cachedTokensIn * (pricing.cachedInputPer1mTokens || 0) / 1000000;
    const inputCost = effectiveInput * pricing.inputPer1mTokens / 1000000;
    const outputCost = tokensOut * pricing.outputPer1mTokens / 1000000;

    const total = inputCost + cachedCost + outputCost;
    if (total < 0) {
      throw new Error(`Negative cost calculated for model ${model}`);
    }
    return Math.round(total * 100000000) / 100000000; // 8 decimal precision
  }

  async recordLLMCall(params: {
    model: string;
    tokensIn: number;
    tokensOut: number;
    cachedTokensIn?: number;
    userId?: string;
    sessionId?: string;
    turnId?: string;
  }): Promise<number> {
    const tokensIn = params.tokensIn;
    const tokensOut = params.tokensOut;
    const cachedTokensIn = params.cachedTokensIn || 0;

    const cost = this.calcLlmCost(params.model, tokensIn, tokensOut, cachedTokensIn);

    const ctx = getContext(false);
    ctx.incrementTokens(tokensIn, tokensOut);
    ctx.incrementCost(cost);

    try {
      await this.store.incrementCost({
        userId: params.userId || ctx.userId,
        turnId: params.turnId || ctx.turnId || Math.random().toString(36).substring(2, 9),
        costUsd: cost,
        operation: 'chat_or_generate_usd',
        service: 'chat',
        model: params.model,
        inputTokens: tokensIn,
        outputTokens: tokensOut,
        sessionId: params.sessionId || ctx.sessionId
      });
    } catch (err: any) {
      logger.warn(Events.COST_TRACKER_DEGRADED, { error: err.message || err });
    }

    return cost;
  }

  async recordEmbedding(params: {
    model: string;
    charCount: number;
    cacheHit?: boolean;
  }): Promise<number> {
    const isHit = params.cacheHit || false;
    let cost = 0.0;
    if (!isHit) {
      const rate = EMBED_COST_PER_1K_CHARS[params.model] || 0.000025;
      cost = Math.round((params.charCount * rate / 1000) * 100000000) / 100000000;
    }

    const ctx = getContext(false);
    ctx.incrementCost(cost);

    try {
      await this.store.incrementCost({
        userId: ctx.userId,
        turnId: ctx.turnId || Math.random().toString(36).substring(2, 9),
        costUsd: cost,
        operation: 'embed_usd',
        service: 'embed',
        model: params.model,
        inputTokens: params.charCount
      });
    } catch (err: any) {
      logger.warn(Events.COST_TRACKER_DEGRADED, { error: err.message || err });
    }

    return cost;
  }
}

/**
 * PostgreSQL-backed cost store.
 */
export class PgCostStore extends BaseCostStore {
  constructor(
    private client: pg.Client | pg.Pool,
    private tableName: string = 'llm_cost_events'
  ) {
    super();
  }

  async initializeSchema(): Promise<void> {
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        service TEXT NOT NULL,
        model TEXT NOT NULL,
        user_id UUID,
        session_id UUID,
        turn_id UUID,
        input_tokens INT NOT NULL DEFAULT 0,
        output_tokens INT NOT NULL DEFAULT 0,
        audio_seconds REAL NOT NULL DEFAULT 0.0,
        images_count INT NOT NULL DEFAULT 0,
        cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0.0,
        coin_revenue_usd NUMERIC(10, 6) NOT NULL DEFAULT 0.0
      );
      CREATE INDEX IF NOT EXISTS idx_cost_events_user_ts ON ${this.tableName} (user_id, timestamp);
      CREATE INDEX IF NOT EXISTS idx_cost_events_timestamp ON ${this.tableName} (timestamp);
    `);
  }

  async checkBudget(userId: string, tier: string, limits: BudgetConfig): Promise<BudgetStatus> {
    const startOfToday = new Date();
    startOfToday.setUTCHours(0, 0, 0, 0);

    const startOfThisMonth = new Date();
    startOfThisMonth.setUTCDate(1);
    startOfThisMonth.setUTCHours(0, 0, 0, 0);

    // Run aggregations
    // Check global daily cost
    const globalRes = await this.client.query(
      `SELECT COALESCE(SUM(cost_usd), 0) as total FROM ${this.tableName} WHERE timestamp >= $1`,
      [startOfToday]
    );
    const globalSpent = parseFloat(globalRes.rows[0].total);

    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD) {
      return BudgetStatus.GLOBAL_HARD_STOP;
    }
    if (globalSpent >= PLATFORM_DAILY_LIMIT_USD * 0.8) {
      return BudgetStatus.GLOBAL_WARNING;
    }

    // Check user daily and monthly cost
    const userRes = await this.client.query(
      `SELECT 
        COALESCE(SUM(CASE WHEN timestamp >= $2 THEN cost_usd ELSE 0 END), 0) as daily,
        COALESCE(SUM(cost_usd), 0) as monthly
       FROM ${this.tableName} 
       WHERE user_id = $1 AND timestamp >= $3`,
      [userId, startOfToday, startOfThisMonth]
    );

    const dailySpent = parseFloat(userRes.rows[0].daily);
    const monthlySpent = parseFloat(userRes.rows[0].monthly);

    if (dailySpent >= limits.dailyLimitUsd || monthlySpent >= limits.monthlyLimitUsd) {
      return BudgetStatus.EXCEEDED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.throttleThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.throttleThresholdPct
    ) {
      return BudgetStatus.THROTTLED;
    }

    if (
      dailySpent >= limits.dailyLimitUsd * limits.alertThresholdPct ||
      monthlySpent >= limits.monthlyLimitUsd * limits.alertThresholdPct
    ) {
      return BudgetStatus.WARNING;
    }

    return BudgetStatus.OK;
  }

  async incrementCost(params: {
    userId: string;
    turnId: string;
    costUsd: number;
    operation: string;
    service: string;
    model: string;
    inputTokens?: number;
    outputTokens?: number;
    sessionId?: string;
  }): Promise<void> {
    const userId = params.userId && params.userId !== 'null' ? params.userId : null;
    const turnId = params.turnId && params.turnId !== 'null' ? params.turnId : null;
    const sessionId = params.sessionId && params.sessionId !== 'null' ? params.sessionId : null;

    await this.client.query(
      `INSERT INTO ${this.tableName} 
       (service, model, user_id, session_id, turn_id, input_tokens, output_tokens, cost_usd)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [
        params.service || 'chat',
        params.model || 'unknown',
        userId,
        sessionId,
        turnId,
        params.inputTokens || 0,
        params.outputTokens || 0,
        params.costUsd || 0.0
      ]
    );
  }

  async getDailyCost(userId: string, date: string): Promise<DailyCostSummary> {
    const startDate = new Date(date);
    const endDate = new Date(date);
    endDate.setDate(endDate.getDate() + 1);

    const res = await this.client.query(
      `SELECT 
        COALESCE(SUM(cost_usd), 0) as total,
        COALESCE(SUM(CASE WHEN service = 'chat' THEN cost_usd ELSE 0 END), 0) as chat,
        COALESCE(SUM(CASE WHEN service = 'live' THEN cost_usd ELSE 0 END), 0) as live,
        COALESCE(SUM(CASE WHEN service = 'image' THEN cost_usd ELSE 0 END), 0) as image,
        COALESCE(SUM(CASE WHEN service = 'embed' THEN cost_usd ELSE 0 END), 0) as embed,
        COALESCE(SUM(CASE WHEN service = 'chat' THEN 1 ELSE 0 END), 0) as turns
       FROM ${this.tableName} 
       WHERE user_id = $1 AND timestamp >= $2 AND timestamp < $3`,
      [userId, startDate, endDate]
    );

    const row = res.rows[0];
    return {
      totalUsd: parseFloat(row.total),
      llmUsd: parseFloat(row.chat),
      liveUsd: parseFloat(row.live),
      imageUsd: parseFloat(row.image),
      imageEditUsd: 0,
      embedUsd: parseFloat(row.embed),
      turnCount: parseInt(row.turns),
      date
    };
  }
}
