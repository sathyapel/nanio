import { BaseModel, Message, LLMResponse, GenerateOptions, ModelTier } from '@nanio/core';

/**
 * Base abstract class for dynamic Model Routers.
 */
export abstract class BaseRouter {
  abstract route(messages: Message[], context?: any): Promise<BaseModel>;
}

/**
 * A virtual BaseModel wrapper that executes automatic switching failover across a list of models.
 */
export class FallbackModel extends BaseModel {
  public name: string;

  constructor(
    public models: BaseModel[],
    name: string = 'fallback-chain'
  ) {
    super();
    this.name = name;
    if (models.length === 0) {
      throw new Error('FallbackModel requires at least one model candidate.');
    }
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    let lastError: Error | null = null;

    for (const model of this.models) {
      try {
        // Attempt generation using candidate model
        return await model.generate(messages, options);
      } catch (error: any) {
        lastError = error;
        console.warn(
          `\x1b[33m[Registry Failover] Model "${model.name}" failed: ${
            error.message || error
          }. Auto-switching to next candidate...\x1b[0m`
        );
      }
    }

    throw new Error(
      `FallbackModel execution failed. All candidates in the chain failed. Last error: ${
        lastError?.message || lastError
      }`
    );
  }
}

export class FallbackRouter extends BaseRouter {
  constructor(private candidates: BaseModel[]) {
    super();
  }

  async route(messages: Message[], context?: any): Promise<BaseModel> {
    return new FallbackModel(this.candidates);
  }
}

export interface ModelEntry {
  tier: ModelTier;
  model: BaseModel;
  name: string;
}

const TIER_PRIORITY: Record<ModelTier, number> = {
  HEAVY: 10,
  PRIMARY: 20,
  LIGHTER: 30,
  CROSS_PROVIDER: 40,
  FREE: 50
};

export class LLMRegistry extends BaseModel {
  public name = 'llm-registry';
  private chain: ModelEntry[] = [];

  constructor(
    entries: ModelEntry[],
    private costStore?: any // Pluggable Cost Store
  ) {
    super();
    this.chain = entries;
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    const config = options?.config;
    let startTier: ModelTier = config?.modelTier || 'PRIMARY';

    // 1. Check budget status if costStore is available
    if (this.costStore && config?.userId) {
      try {
        const budgetStatus = await this.costStore.checkBudget(config.userId, config.userTier || 'FREE');

        if (budgetStatus === 'THROTTLED') {
          startTier = 'LIGHTER';
        }
      } catch (err) {
        // Fall back to config/default
      }
    }

    const attempted: string[] = [];
    const reasons: string[] = [];

    for (const entry of this.chain) {
      // Skip tiers that are too heavy for startTier
      const entryPriority = TIER_PRIORITY[entry.tier] || 99;
      const startPriority = TIER_PRIORITY[startTier] || 99;
      if (entryPriority < startPriority) {
        continue;
      }

      attempted.push(entry.name);
      try {
        const response = await entry.model.generate(messages, options);
        return response;
      } catch (error: any) {
        // Fall back only on RateLimitError or CircuitOpenError
        const errorName = error.name || error.constructor?.name;
        if (errorName === 'RateLimitError' || errorName === 'CircuitOpenError') {
          reasons.push(errorName);
          console.warn(
            `\x1b[33m[Registry Fallback] Model "${entry.name}" (${entry.tier}) failed with ${errorName}. Falling back...\x1b[0m`
          );
          continue;
        }
        // Rethrow non-fallback error immediately
        throw error;
      }
    }

    throw new Error(
      `All providers failed. Tried: ${attempted.join(', ')}. Reasons: ${reasons.join(', ')}`
    );
  }
}

