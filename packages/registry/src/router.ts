import { BaseModel, Message, LLMResponse, GenerateOptions } from '@nanio/core';

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

/**
 * A router that returns a FallbackModel to support automated model switching.
 */
export class FallbackRouter extends BaseRouter {
  constructor(private candidates: BaseModel[]) {
    super();
  }

  async route(messages: Message[], context?: any): Promise<BaseModel> {
    return new FallbackModel(this.candidates);
  }
}
