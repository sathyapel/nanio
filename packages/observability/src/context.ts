import { AsyncLocalStorage } from 'async_hooks';

export class ContextNotInitialized extends Error {
  constructor(message?: string) {
    super(message || 'RequestContext is not initialized. Ensure context is set before accessing.');
    this.name = 'ContextNotInitialized';
  }
}

/**
 * Carries trace, session, identity, token, and cost metrics for a single request lifecycle.
 */
export class RequestContext {
  traceId: string;
  turnId: string;
  sessionId: string;
  userId: string;
  stage: string;
  tier: string;
  totalTokensIn: number;
  totalTokensOut: number;
  totalCostUsd: number;
  toolCallCount: number;
  modelUsed: string;
  fallbackFired: boolean;

  constructor(data?: Partial<RequestContext>) {
    this.traceId = data?.traceId || Math.random().toString(36).substring(2, 15);
    this.turnId = data?.turnId || '';
    this.sessionId = data?.sessionId || '';
    this.userId = data?.userId || '';
    this.stage = data?.stage || '';
    this.tier = data?.tier || 'FREE';
    this.totalTokensIn = data?.totalTokensIn || 0;
    this.totalTokensOut = data?.totalTokensOut || 0;
    this.totalCostUsd = data?.totalCostUsd || 0.0;
    this.toolCallCount = data?.toolCallCount || 0;
    this.modelUsed = data?.modelUsed || '';
    this.fallbackFired = data?.fallbackFired || false;
  }

  /**
   * Serialize context to a plain JSON object for passing into background queues.
   */
  snapshot(): Record<string, any> {
    return {
      traceId: this.traceId,
      turnId: this.turnId,
      sessionId: this.sessionId,
      userId: this.userId,
      stage: this.stage,
      tier: this.tier,
      totalTokensIn: this.totalTokensIn,
      totalTokensOut: this.totalTokensOut,
      totalCostUsd: this.totalCostUsd,
      toolCallCount: this.toolCallCount,
      modelUsed: this.modelUsed,
      fallbackFired: this.fallbackFired
    };
  }

  incrementTokens(inTokens: number, outTokens: number): void {
    this.totalTokensIn += inTokens;
    this.totalTokensOut += outTokens;
  }

  incrementCost(usd: number): void {
    this.totalCostUsd += usd;
  }

  incrementToolCalls(count: number = 1): void {
    this.toolCallCount += count;
  }
}

// Global AsyncLocalStorage store for Node.js context propagation
const contextStorage = new AsyncLocalStorage<RequestContext>();

/**
 * Run a callback function within a specific RequestContext.
 */
export function runWithContext<T>(ctx: RequestContext, callback: () => T): T {
  return contextStorage.run(ctx, callback);
}

/**
 * Get the current request context from storage.
 * @param strict If true, throws ContextNotInitialized if context is empty. Otherwise, returns a default mock context.
 */
export function getContext(strict: boolean = true): RequestContext {
  const ctx = contextStorage.getStore();
  if (!ctx) {
    if (strict) {
      throw new ContextNotInitialized(
        'RequestContext is not set. Ensure runWithContext() is executed before calling client methods.'
      );
    }
    return new RequestContext();
  }
  return ctx;
}

/**
 * Factory helper to build a fresh RequestContext.
 */
export function newRequestContext(params?: {
  userId?: string;
  sessionId?: string;
  stage?: string;
  tier?: string;
  traceId?: string;
}): RequestContext {
  return new RequestContext({
    traceId: params?.traceId,
    userId: params?.userId,
    sessionId: params?.sessionId,
    stage: params?.stage,
    tier: params?.tier
  });
}

/**
 * Restores a serialized request context snapshot to run a background worker function.
 */
export function restoreContext<T>(snapshot: Record<string, any>, callback: () => T): T {
  const ctx = new RequestContext(snapshot);
  return runWithContext(ctx, callback);
}

/**
 * Helper context manager to execute test cases with mock variables.
 */
export function testContext<T>(
  callback: () => T,
  params?: { userId?: string; stage?: string }
): T {
  const ctx = newRequestContext({
    userId: params?.userId || 'test-user',
    stage: params?.stage || 'STRANGER',
    tier: 'FREE',
    traceId: 'test-trace-id'
  });
  return runWithContext(ctx, callback);
}
