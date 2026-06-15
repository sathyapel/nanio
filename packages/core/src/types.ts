export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

export interface Message {
  role: MessageRole;
  content: string;
  name?: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface LLMResponse {
  content: string;
  toolCalls?: ToolCall[];
  usage?: TokenUsage;
  raw?: any;
}

export interface GenerateOptions {
  temperature?: number;
  maxTokens?: number;
  stopSequences?: string[];
  tools?: any[]; // Serialized JSON schemas of tools
  config?: ChatConfig;
}

export type UserTier = 'FREE' | 'PRO' | 'PREMIUM' | 'VIP' | 'CREATOR' | 'OPERATOR';
export type ModelTier = 'HEAVY' | 'PRIMARY' | 'LIGHTER' | 'FREE' | 'CROSS_PROVIDER';

export class ChatConfig {
  public temperature?: number;
  public maxTokens?: number;
  public thinkingLevel?: 'NONE' | 'MINIMAL' | 'AUTO';
  public modelTier?: ModelTier;
  public jsonMode?: boolean;
  public tools?: any[];
  public toolChoice?: string;
  public maxToolIterations?: number;
  public maxToolCallsPerTurn?: number;
  public toolTimeoutSeconds?: number;
  public maxToolConcurrency?: number;
  public contextTokenBudget?: number;
  public maxToolResultBytes?: number;
  public userId?: string | null;
  public userTier?: UserTier;

  constructor(init?: Partial<ChatConfig>) {
    Object.assign(this, init);
  }

  /**
   * Helper to merge/override properties in place.
   */
  update(updates: Partial<ChatConfig>): void {
    Object.assign(this, updates);
  }

  /**
   * Construct a ChatConfig directly from an LLMConfig record from the database.
   */
  static fromLLMConfig(dbConfig: {
    user_tier: string;
    model_tier: string;
    temperature: number;
    max_output_tokens: number;
    context_token_budget: number;
  }): ChatConfig {
    return new ChatConfig({
      userTier: dbConfig.user_tier as UserTier,
      modelTier: dbConfig.model_tier as ModelTier,
      temperature: dbConfig.temperature,
      maxTokens: dbConfig.max_output_tokens,
      contextTokenBudget: dbConfig.context_token_budget
    });
  }
}

