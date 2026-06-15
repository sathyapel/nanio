import pg from 'pg';
import { BaseMemory } from './memory.js';
import { Message, MessageRole } from './types.js';

export interface LLMConfig {
  user_tier: string;
  model_tier: string;
  model_name_override: string | null;
  history_limit: number;
  context_token_budget: number;
  daily_limit_usd: number | null;
  monthly_limit_usd: number | null;
  temperature: number;
  max_output_tokens: number;
  memory_importance_threshold: number;
}

export const LLMCONFIG_DEFAULTS: Record<string, Omit<LLMConfig, 'user_tier'>> = {
  FREE: {
    model_tier: 'FREE',
    model_name_override: null,
    history_limit: 10,
    context_token_budget: 300000,
    daily_limit_usd: 0.10,
    monthly_limit_usd: 2.00,
    temperature: 0.9,
    max_output_tokens: 8192,
    memory_importance_threshold: 0.6,
  },
  PRO: {
    model_tier: 'LIGHTER',
    model_name_override: null,
    history_limit: 15,
    context_token_budget: 600000,
    daily_limit_usd: 0.50,
    monthly_limit_usd: 10.00,
    temperature: 0.9,
    max_output_tokens: 30000,
    memory_importance_threshold: 0.55,
  },
  PREMIUM: {
    model_tier: 'PRIMARY',
    model_name_override: null,
    history_limit: 20,
    context_token_budget: 900000,
    daily_limit_usd: 2.00,
    monthly_limit_usd: 40.00,
    temperature: 0.9,
    max_output_tokens: 60000,
    memory_importance_threshold: 0.5,
  },
  VIP: {
    model_tier: 'HEAVY',
    model_name_override: null,
    history_limit: 20,
    context_token_budget: 900000,
    daily_limit_usd: 5.00,
    monthly_limit_usd: 100.00,
    temperature: 0.95,
    max_output_tokens: 60000,
    memory_importance_threshold: 0.45,
  },
  CREATOR: {
    model_tier: 'PRIMARY',
    model_name_override: null,
    history_limit: 20,
    context_token_budget: 900000,
    daily_limit_usd: 2.00,
    monthly_limit_usd: 40.00,
    temperature: 0.9,
    max_output_tokens: 60000,
    memory_importance_threshold: 0.5,
  },
  OPERATOR: {
    model_tier: 'HEAVY',
    model_name_override: null,
    history_limit: 20,
    context_token_budget: 900000,
    daily_limit_usd: null,
    monthly_limit_usd: null,
    temperature: 0.9,
    max_output_tokens: 60000,
    memory_importance_threshold: 0.4,
  },
};

export class PgLLMConfigRepository {
  constructor(
    private client: pg.Client | pg.Pool,
    private tableName: string = 'llm_configs',
    private redis?: any
  ) {}

  async initializeSchema(): Promise<void> {
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        user_tier TEXT PRIMARY KEY,
        model_tier TEXT NOT NULL,
        model_name_override TEXT,
        history_limit INT NOT NULL DEFAULT 20,
        context_token_budget INT NOT NULL DEFAULT 900000,
        daily_limit_usd NUMERIC(10, 4),
        monthly_limit_usd NUMERIC(10, 4),
        temperature REAL NOT NULL DEFAULT 0.9,
        max_output_tokens INT NOT NULL DEFAULT 60000,
        memory_importance_threshold REAL NOT NULL DEFAULT 0.5
      );
    `);
  }

  private cacheKey(tier: string): string {
    return `llmconfig:tier:${tier}`;
  }

  async getForTier(tier: string): Promise<LLMConfig> {
    const key = this.cacheKey(tier);

    if (this.redis) {
      try {
        const cachedRaw = await this.redis.get(key);
        if (cachedRaw) {
          const data = JSON.parse(cachedRaw);
          return {
            user_tier: tier,
            ...data
          };
        }
      } catch (err) {
        // Fall through
      }
    }

    try {
      const res = await this.client.query(
        `SELECT * FROM ${this.tableName} WHERE user_tier = $1`,
        [tier]
      );
      if (res.rows.length > 0) {
        const row = res.rows[0];
        const data = {
          model_tier: row.model_tier,
          model_name_override: row.model_name_override,
          history_limit: parseInt(row.history_limit),
          context_token_budget: parseInt(row.context_token_budget),
          daily_limit_usd: row.daily_limit_usd ? parseFloat(row.daily_limit_usd) : null,
          monthly_limit_usd: row.monthly_limit_usd ? parseFloat(row.monthly_limit_usd) : null,
          temperature: parseFloat(row.temperature),
          max_output_tokens: parseInt(row.max_output_tokens),
          memory_importance_threshold: parseFloat(row.memory_importance_threshold)
        };

        if (this.redis) {
          try {
            await this.redis.set(key, JSON.stringify(data), 'EX', 300);
          } catch (err) {
            // Ignore
          }
        }

        return {
          user_tier: tier,
          ...data
        };
      }
    } catch (err) {
      // Fall through
    }

    const defaultData = LLMCONFIG_DEFAULTS[tier] || LLMCONFIG_DEFAULTS.FREE;
    return {
      user_tier: tier,
      ...defaultData
    };
  }

  async invalidateCache(tier?: string): Promise<void> {
    if (!this.redis) return;
    if (tier) {
      await this.redis.del(this.cacheKey(tier));
    } else {
      for (const t of Object.keys(LLMCONFIG_DEFAULTS)) {
        await this.redis.del(this.cacheKey(t));
      }
    }
  }
}

export class PgChatMemory extends BaseMemory {
  constructor(
    private client: pg.Client | pg.Pool,
    private sessionId: string,
    private tableName: string = 'conversation_turns',
    private userId: string | null = null
  ) {
    super();
  }

  async initializeSchema(): Promise<void> {
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        turn_id TEXT UNIQUE,
        session_id UUID NOT NULL,
        user_id UUID,
        session_turn_index INT NOT NULL DEFAULT 0,
        input_modality TEXT NOT NULL DEFAULT 'TEXT',
        user_message TEXT NOT NULL,
        ai_response TEXT NOT NULL,
        stage_at_turn TEXT NOT NULL DEFAULT 'STRANGER',
        tokens_in INT NOT NULL DEFAULT 0,
        tokens_out INT NOT NULL DEFAULT 0,
        cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0.0,
        first_token_ms INT,
        total_latency_ms INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        is_delivered BOOLEAN NOT NULL DEFAULT FALSE
      );
      CREATE INDEX IF NOT EXISTS idx_turns_session_idx ON ${this.tableName} (session_id, session_turn_index);
    `);
  }

  async addMessage(message: Message): Promise<void> {
    const userIdVal = this.userId && this.userId !== 'null' ? this.userId : null;

    if (message.role === 'user') {
      const turnId = 'turn_' + Math.random().toString(36).substring(2, 9);
      const indexRes = await this.client.query(
        `SELECT COALESCE(MAX(session_turn_index), 0) as max_idx FROM ${this.tableName} WHERE session_id = $1`,
        [this.sessionId]
      );
      const nextIndex = parseInt(indexRes.rows[0].max_idx) + 1;

      await this.client.query(
        `INSERT INTO ${this.tableName} 
         (turn_id, session_id, user_id, session_turn_index, user_message, ai_response, stage_at_turn)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [turnId, this.sessionId, userIdVal, nextIndex, message.content, '', 'STRANGER']
      );
    } else if (message.role === 'assistant') {
      const latestRes = await this.client.query(
        `SELECT id FROM ${this.tableName} WHERE session_id = $1 ORDER BY session_turn_index DESC LIMIT 1`,
        [this.sessionId]
      );
      
      if (latestRes.rows.length > 0) {
        const id = latestRes.rows[0].id;
        await this.client.query(
          `UPDATE ${this.tableName} SET ai_response = $1 WHERE id = $2`,
          [message.content, id]
        );
      } else {
        const turnId = 'turn_' + Math.random().toString(36).substring(2, 9);
        await this.client.query(
          `INSERT INTO ${this.tableName} 
           (turn_id, session_id, user_id, session_turn_index, user_message, ai_response, stage_at_turn)
           VALUES ($1, $2, $3, 1, $4, $5, $6)`,
          [turnId, this.sessionId, userIdVal, '', message.content, 'STRANGER']
        );
      }
    } else {
      const turnId = 'turn_' + Math.random().toString(36).substring(2, 9);
      const indexRes = await this.client.query(
        `SELECT COALESCE(MAX(session_turn_index), 0) as max_idx FROM ${this.tableName} WHERE session_id = $1`,
        [this.sessionId]
      );
      const nextIndex = parseInt(indexRes.rows[0].max_idx) + 1;
      await this.client.query(
        `INSERT INTO ${this.tableName} 
         (turn_id, session_id, user_id, session_turn_index, user_message, ai_response, stage_at_turn)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [turnId, this.sessionId, userIdVal, nextIndex, `[${message.role}] ${message.content}`, '', 'STRANGER']
      );
    }
  }

  async getMessages(): Promise<Message[]> {
    const res = await this.client.query(
      `SELECT user_message, ai_response FROM ${this.tableName} WHERE session_id = $1 ORDER BY session_turn_index ASC`,
      [this.sessionId]
    );

    const messages: Message[] = [];
    for (const row of res.rows) {
      if (row.user_message) {
        let content = row.user_message;
        let role: MessageRole = 'user';
        const systemMatch = content.match(/^\[(system|tool)\] (.*)/);
        if (systemMatch) {
          role = systemMatch[1] as MessageRole;
          content = systemMatch[2];
        }
        messages.push({ role, content });
      }
      if (row.ai_response) {
        messages.push({ role: 'assistant', content: row.ai_response });
      }
    }
    return messages;
  }

  async clear(): Promise<void> {
    await this.client.query(
      `DELETE FROM ${this.tableName} WHERE session_id = $1`,
      [this.sessionId]
    );
  }
}
