import { z } from 'zod';
import { ToolContext, ToolHooks } from './hooks.js';

/**
 * Utility to convert Zod schema definitions to clean JSON Schema
 */
export function zodToJsonSchema(schema: z.ZodTypeAny): any {
  if (schema instanceof z.ZodObject) {
    const properties: Record<string, any> = {};
    const required: string[] = [];
    const shape = schema.shape;
    for (const [key, value] of Object.entries(shape)) {
      properties[key] = zodToJsonSchema(value as z.ZodTypeAny);
      if (!(value instanceof z.ZodOptional) && !(value instanceof z.ZodNullable)) {
        required.push(key);
      }
    }
    return {
      type: 'object',
      properties,
      required: required.length > 0 ? required : undefined,
    };
  }

  if (schema instanceof z.ZodString) {
    return { type: 'string', description: schema.description };
  }
  if (schema instanceof z.ZodNumber) {
    return { type: 'number', description: schema.description };
  }
  if (schema instanceof z.ZodBoolean) {
    return { type: 'boolean', description: schema.description };
  }
  if (schema instanceof z.ZodArray) {
    return {
      type: 'array',
      items: zodToJsonSchema(schema.element),
      description: schema.description,
    };
  }
  if (schema instanceof z.ZodOptional || schema instanceof z.ZodNullable) {
    return zodToJsonSchema(schema.unwrap());
  }
  if (schema instanceof z.ZodEnum) {
    return {
      type: 'string',
      enum: schema.options,
      description: schema.description,
    };
  }

  // Fallback
  return { type: 'string' };
}

/**
 * Base abstract Tool class. All tools in nanio inherit from this class.
 */
export abstract class BaseTool<TSchema extends z.ZodObject<any> = z.ZodObject<any>, TOutput = any> {
  abstract name: string;
  abstract description: string;
  abstract schema: TSchema;

  hooks?: ToolHooks<z.infer<TSchema>, TOutput>;

  protected abstract execute(input: z.infer<TSchema>, context: ToolContext): Promise<TOutput>;

  /**
   * Compiles the tool definition and parameter schema to a standard format for LLMs
   */
  toJSONSchema(): any {
    return {
      name: this.name,
      description: this.description,
      parameters: zodToJsonSchema(this.schema),
    };
  }

  /**
   * Run the tool: validates inputs, executes hooks, performs work, handles failures
   */
  async run(input: unknown, metadata?: Record<string, any>): Promise<TOutput> {
    const context: ToolContext = {
      toolName: this.name,
      timestamp: Date.now(),
      metadata,
    };

    try {
      // 1. Runtime validation
      const validated = this.schema.parse(input);

      // 2. Before execution hook
      let processedInput = validated;
      if (this.hooks?.beforeCall) {
        processedInput = await this.hooks.beforeCall(validated, context);
      }

      // 3. Execution
      let output = await this.execute(processedInput, context);

      // 4. After execution hook
      if (this.hooks?.afterCall) {
        output = await this.hooks.afterCall(output, context);
      }

      return output;
    } catch (error: any) {
      // 5. Error hook for recovery or logging
      if (this.hooks?.onCallError) {
        return await this.hooks.onCallError(error, context);
      }
      throw error;
    }
  }
}

/**
 * A concrete wrapper that dynamically creates tools from standard functions.
 */
export class FunctionTool<TSchema extends z.ZodObject<any> = z.ZodObject<any>, TOutput = any> extends BaseTool<TSchema, TOutput> {
  constructor(
    public name: string,
    public description: string,
    public schema: TSchema,
    private handler: (input: z.infer<TSchema>, context: ToolContext) => Promise<TOutput> | TOutput,
    hooks?: ToolHooks<z.infer<TSchema>, TOutput>
  ) {
    super();
    this.hooks = hooks;
  }

  protected async execute(input: z.infer<TSchema>, context: ToolContext): Promise<TOutput> {
    return this.handler(input, context);
  }
}
