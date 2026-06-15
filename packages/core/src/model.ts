import { Message, LLMResponse, GenerateOptions } from './types.js';

/**
 * Base abstract Model class. Custom LLM providers (e.g. OpenAI, Anthropic, Gemini, local Ollama) inherit from this.
 */
export abstract class BaseModel {
  abstract name: string;
  abstract generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse>;
}
