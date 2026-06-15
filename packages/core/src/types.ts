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
}
