import { BaseModel, Message, LLMResponse, GenerateOptions, ChatConfig } from '@nanio/core';
import { getContext, getLogger, Events } from '@nanio/observability';

const logger = getLogger('BaseAIClient');

export class CircuitOpenError extends Error {
  constructor(public clientName: string, public remainingSeconds: number) {
    super(`Circuit breaker for ${clientName} is OPEN. Retrying in ${remainingSeconds.toFixed(1)}s.`);
    this.name = 'CircuitOpenError';
  }
}

export class RateLimitError extends Error {
  constructor(public attempts: number) {
    super(`Rate limit exceeded after ${attempts} attempts.`);
    this.name = 'RateLimitError';
  }
}

export enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN'
}

/**
 * Simple async circuit breaker.
 */
export class CircuitBreaker {
  private state = CircuitState.CLOSED;
  private failureCount = 0;
  private openedAt: number | null = null;

  constructor(
    private failureThreshold: number = 5,
    private cooldownSeconds: number = 30,
    private clientName: string = 'unknown',
    private onCircuitOpen?: (failureCount: number, cooldownSeconds: number) => Promise<void>
  ) {}

  async check(): Promise<void> {
    if (this.state === CircuitState.OPEN) {
      const elapsed = (Date.now() - (this.openedAt || 0)) / 1000;
      if (elapsed >= this.cooldownSeconds) {
        this.state = CircuitState.HALF_OPEN;
        logger.warn(Events.CLIENT_CIRCUIT_HALF_OPEN, {
          client: this.clientName,
          elapsed_seconds: Math.round(elapsed)
        });
      } else {
        const remaining = this.cooldownSeconds - elapsed;
        throw new CircuitOpenError(this.clientName, remaining);
      }
    }
  }

  async recordSuccess(): Promise<void> {
    if (this.state === CircuitState.HALF_OPEN) {
      logger.info(Events.CLIENT_CIRCUIT_CLOSED, { client: this.clientName });
    }
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.openedAt = null;
  }

  async recordFailure(): Promise<void> {
    this.failureCount += 1;
    if (this.state === CircuitState.HALF_OPEN || this.failureCount >= this.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.openedAt = Date.now();
      logger.error(Events.CLIENT_CIRCUIT_OPEN, {
        client: this.clientName,
        failure_count: this.failureCount,
        cooldown_seconds: this.cooldownSeconds
      });
      if (this.onCircuitOpen) {
        await this.onCircuitOpen(this.failureCount, this.cooldownSeconds);
      }
    }
  }
}

/**
 * Token bucket rate limiter with wait thresholds and automatic model switching triggers.
 */
export class RateLimiter {
  private tokens: number;
  private lastRefill = performance.now();

  constructor(
    private maxRate: number,
    private timePeriodSeconds: number = 1.0,
    private maxWaitSeconds: number = 5.0,
    private clientName: string = 'unknown',
    private onWait?: (waitSeconds: number) => Promise<void>
  ) {
    this.tokens = maxRate;
  }

  async acquire(): Promise<number> {
    const now = performance.now();
    const elapsed = (now - this.lastRefill) / 1000;

    const refill = elapsed * (this.maxRate / this.timePeriodSeconds);
    this.tokens = Math.min(this.maxRate, this.tokens + refill);
    this.lastRefill = now;

    // Claim one token
    this.tokens -= 1.0;

    if (this.tokens >= 0.0) {
      return 0.0;
    }

    const waitSeconds = Math.abs(this.tokens) * (this.timePeriodSeconds / this.maxRate);

    if (waitSeconds > this.maxWaitSeconds) {
      // Refund token so we don't block subsequent slots
      this.tokens += 1.0;
      logger.warn(Events.COST_BUDGET_THROTTLED, {
        client: this.clientName,
        wait_seconds: waitSeconds,
        limit_seconds: this.maxWaitSeconds
      });
      throw new RateLimitError(1);
    }

    if (waitSeconds > 0.0) {
      if (this.onWait) {
        await this.onWait(waitSeconds);
      } else {
        logger.info(Events.CLIENT_RATE_LIMIT_WAIT, {
          client: this.clientName,
          wait_ms: Math.round(waitSeconds * 1000)
        });
      }
      await new Promise(resolve => setTimeout(resolve, waitSeconds * 1000));
    }

    return waitSeconds;
  }
}

/**
 * Shared resilient core abstract client class for all provider models.
 */
export abstract class BaseAIClient extends BaseModel {
  abstract name: string;
  abstract generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse>;
  abstract countTokens(messages: string | Message[]): Promise<number>;

  async chatComplete(messages: Message[], options?: GenerateOptions): Promise<string> {
    const res = await this.generate(messages, options);
    return res.content;
  }

  async chatJson(messages: Message[], options?: GenerateOptions): Promise<any> {
    const config = options?.config || new ChatConfig();
    config.jsonMode = true;
    const res = await this.generate(messages, {
      ...options,
      config
    });
    try {
      return JSON.parse(res.content);
    } catch (err) {
      return { content: res.content };
    }
  }

  protected circuit: CircuitBreaker;
  protected rateLimiters: Record<string, RateLimiter>;

  protected maxRetryAttempts: number = 3;
  protected baseRetryDelaySeconds: number = 1.0;

  constructor(
    clientName: string,
    protected uiPublisher?: any,
    protected perfTracker?: any
  ) {
    super();
    this.circuit = new CircuitBreaker(5, 30, clientName, async (failCount, cooldown) => {
      if (this.perfTracker) {
        await this.perfTracker.incrementCircuitOpen(clientName);
      }
    });

    // Limiters for all 6 tiers
    this.rateLimiters = {
      VIP: new RateLimiter(25.0, 1.0, 1.0, clientName, async (wait) => this.onRateLimitWait(wait)),
      OPERATOR: new RateLimiter(25.0, 1.0, 1.0, clientName, async (wait) => this.onRateLimitWait(wait)),
      PREMIUM: new RateLimiter(15.0, 1.0, 3.0, clientName, async (wait) => this.onRateLimitWait(wait)),
      CREATOR: new RateLimiter(15.0, 1.0, 3.0, clientName, async (wait) => this.onRateLimitWait(wait)),
      PRO: new RateLimiter(12.0, 1.0, 5.0, clientName, async (wait) => this.onRateLimitWait(wait)),
      FREE: new RateLimiter(10.0, 1.0, 8.0, clientName, async (wait) => this.onRateLimitWait(wait))
    };
  }

  protected async onRateLimitWait(waitSeconds: number): Promise<void> {
    logger.info(Events.CLIENT_RATE_LIMIT_WAIT, {
      client: this.name,
      wait_ms: Math.round(waitSeconds * 1000)
    });

    if (waitSeconds > 5.0 && this.uiPublisher) {
      const ctx = getContext(false);
      const userId = ctx.userId;
      if (userId) {
        await this.uiPublisher.sendTypingIndicator(userId, waitSeconds);
      }
    }
  }

  /**
   * Wrapper executing API call factory with retry logic and full jitter backoff.
   */
  protected async withRetry<T>(coroFactory: () => Promise<T>): Promise<T> {
    const ctx = getContext(false);
    const tier = ctx.tier || 'FREE';
    const limiter = this.rateLimiters[tier] || this.rateLimiters.FREE;

    let delay = this.baseRetryDelaySeconds;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetryAttempts; attempt++) {
      // 1. Check circuit breaker state
      await this.circuit.check();

      // 2. Enforce rate limit bucket waits
      await limiter.acquire();

      try {
        const result = await coroFactory();
        await this.circuit.recordSuccess();
        return result;
      } catch (err: any) {
        lastError = err;
        await this.circuit.recordFailure();

        const status = err.status || err.statusCode || (err.message && err.message.match(/status (\d+)/)?.[1]);
        const statusCode = status ? parseInt(status) : null;

        // Fail fast on non-retryable errors
        if (statusCode && [400, 401, 403, 404].includes(statusCode)) {
          throw err;
        }

        logger.warn(Events.CLIENT_RETRY, {
          client: this.name,
          attempt: attempt + 1,
          error: err.message || err,
          status_code: statusCode
        });

        if (this.perfTracker) {
          await this.perfTracker.incrementRetry(this.name);
        }

        if (attempt < this.maxRetryAttempts - 1) {
          // Full jitter backoff sleep
          const cap = 60.0;
          const exponentialDelay = delay * Math.pow(2, attempt);
          const jitterDelay = Math.random() * Math.min(cap, exponentialDelay);
          await new Promise(resolve => setTimeout(resolve, jitterDelay * 1000));
        }
      }
    }

    throw lastError || new Error('Request execution failed.');
  }
}
