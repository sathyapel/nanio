export interface ToolContext {
  toolName: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface ToolHooks<TInput = any, TOutput = any> {
  beforeCall?: (input: TInput, context: ToolContext) => Promise<TInput> | TInput;
  afterCall?: (output: TOutput, context: ToolContext) => Promise<TOutput> | TOutput;
  onCallError?: (error: Error, context: ToolContext) => Promise<TOutput> | TOutput;
}
