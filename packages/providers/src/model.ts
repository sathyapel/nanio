import { Message, LLMResponse, GenerateOptions, ToolCall } from '@nanio/core';
import { BaseAIClient } from './base_client.js';

/**
 * Recursively converts lower-case schema types (zod defaults) to upper-case (required by Gemini API).
 */
function convertSchemaToGemini(schema: any): any {
  if (!schema) return schema;
  const copy = JSON.parse(JSON.stringify(schema));

  function traverse(obj: any) {
    if (obj && typeof obj === 'object') {
      if (typeof obj.type === 'string') {
        obj.type = obj.type.toUpperCase();
      }
      for (const key of Object.keys(obj)) {
        traverse(obj[key]);
      }
    }
  }

  traverse(copy);
  return copy;
}

/**
 * Maps the standard nanio message format to Gemini contents schema
 */
function mapMessagesToGemini(messages: Message[]): any[] {
  return messages.map(msg => {
    let role: string = msg.role;
    if (role === 'assistant') role = 'model';
    if (role === 'tool') role = 'function';

    const parts: any[] = [];

    if (msg.toolCalls && msg.toolCalls.length > 0) {
      for (const tc of msg.toolCalls) {
        parts.push({
          functionCall: {
            name: tc.name,
            args: tc.arguments
          }
        });
      }
    } else if (msg.role === 'tool') {
      let parsedResponse: any;
      try {
        parsedResponse = JSON.parse(msg.content);
      } catch {
        parsedResponse = { result: msg.content };
      }
      parts.push({
        functionResponse: {
          name: msg.name || '',
          response: typeof parsedResponse === 'object' && parsedResponse !== null ? parsedResponse : { result: parsedResponse }
        }
      });
    } else {
      parts.push({ text: msg.content });
    }

    return { role, parts };
  });
}

/**
 * Adapter for Gemini models using native fetch and resilient BaseAIClient.
 */
export class GeminiModel extends BaseAIClient {
  public name: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'gemini-1.5-flash',
    options?: { apiKey?: string; baseUrl?: string; uiPublisher?: any; perfTracker?: any }
  ) {
    super(modelName, options?.uiPublisher, options?.perfTracker);
    this.name = modelName;
    this.apiKey = options?.apiKey || process.env.GEMINI_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://generativelanguage.googleapis.com';
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    if (!this.apiKey) {
      throw new Error(`Gemini API key is missing. Set GEMINI_API_KEY env variable or pass it directly.`);
    }

    return this.withRetry(async () => {
      const geminiMessages = mapMessagesToGemini(messages);
      const url = `${this.baseUrl}/v1beta/models/${this.name}:generateContent?key=${this.apiKey}`;

      const body: any = {
        contents: geminiMessages,
        generationConfig: {
          temperature: options?.temperature ?? 0.7,
          maxOutputTokens: options?.maxTokens ?? 2048,
          stopSequences: options?.stopSequences
        }
      };

      if (options?.tools && options.tools.length > 0) {
        body.tools = [
          {
            functionDeclarations: options.tools.map(tool => ({
              name: tool.name,
              description: tool.description,
              parameters: convertSchemaToGemini(tool.parameters)
            }))
          }
        ];
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errObj = new Error(`Gemini API call failed with status ${response.status}: ${errorText}`);
        (errObj as any).status = response.status;
        throw errObj;
      }

      const result = await response.json();
      const candidate = result.candidates?.[0];
      if (!candidate) {
        throw new Error(`Gemini returned an empty response. Raw result: ${JSON.stringify(result)}`);
      }

      const parts = candidate.content?.parts || [];
      let text = '';
      const toolCalls: ToolCall[] = [];

      for (const part of parts) {
        if (part.text) {
          text += part.text;
        }
        if (part.functionCall) {
          toolCalls.push({
            id: Math.random().toString(36).substring(2, 9),
            name: part.functionCall.name,
            arguments: part.functionCall.args || {}
          });
        }
      }

      const promptTokens = result.usageMetadata?.promptTokenCount ?? 0;
      const completionTokens = result.usageMetadata?.candidatesTokenCount ?? 0;
      const totalTokens = result.usageMetadata?.totalTokenCount ?? 0;

      return {
        content: text,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: { promptTokens, completionTokens, totalTokens },
        raw: result
      };
    });
  }

  async countTokens(messages: string | Message[]): Promise<number> {
    if (!this.apiKey) {
      throw new Error(`Gemini API key is missing.`);
    }
    const geminiMessages = typeof messages === 'string'
      ? [{ role: 'user', parts: [{ text: messages }] }]
      : mapMessagesToGemini(messages);

    const url = `${this.baseUrl}/v1beta/models/${this.name}:countTokens?key=${this.apiKey}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: geminiMessages })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini countTokens call failed: ${errorText}`);
    }

    const result = await response.json();
    return result.totalTokens ?? 0;
  }
}

/**
 * Adapter for OpenAI models using native fetch and resilient BaseAIClient.
 */
export class OpenAIModel extends BaseAIClient {
  public name: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'gpt-4o-mini',
    options?: { apiKey?: string; baseUrl?: string; uiPublisher?: any; perfTracker?: any }
  ) {
    super(modelName, options?.uiPublisher, options?.perfTracker);
    this.name = modelName;
    this.apiKey = options?.apiKey || process.env.OPENAI_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://api.openai.com/v1';
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    if (!this.apiKey) {
      throw new Error(`OpenAI API key is missing. Set OPENAI_API_KEY env variable or pass it directly.`);
    }

    return this.withRetry(async () => {
      const url = `${this.baseUrl}/chat/completions`;

      // Map message list to OpenAI structure
      const openaiMessages = messages.map(msg => {
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          return {
            role: 'assistant',
            content: msg.content || null,
            tool_calls: msg.toolCalls.map(tc => ({
              id: tc.id,
              type: 'function',
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.arguments)
              }
            }))
          };
        }

        if (msg.role === 'tool') {
          return {
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content: msg.content
          };
        }

        return {
          role: msg.role,
          content: msg.content
        };
      });

      const body: any = {
        model: this.name,
        messages: openaiMessages,
        temperature: options?.temperature ?? 0.7,
        max_tokens: options?.maxTokens,
        stop: options?.stopSequences
      };

      if (options?.tools && options.tools.length > 0) {
        body.tools = options.tools.map(tool => ({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters
          }
        }));
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errObj = new Error(`OpenAI API call failed with status ${response.status}: ${errorText}`);
        (errObj as any).status = response.status;
        throw errObj;
      }

      const result = await response.json();
      const choice = result.choices?.[0];
      if (!choice) {
        throw new Error(`OpenAI returned an empty response. Raw result: ${JSON.stringify(result)}`);
      }

      const text = choice.message?.content || '';
      const toolCalls: ToolCall[] = [];

      if (choice.message?.tool_calls) {
        for (const tc of choice.message.tool_calls) {
          if (tc.type === 'function') {
            let args: Record<string, any> = {};
            try {
              args = JSON.parse(tc.function.arguments || '{}');
            } catch {
              args = { _raw: tc.function.arguments };
            }
            toolCalls.push({
              id: tc.id,
              name: tc.function.name,
              arguments: args
            });
          }
        }
      }

      const promptTokens = result.usage?.prompt_tokens ?? 0;
      const completionTokens = result.usage?.completion_tokens ?? 0;
      const totalTokens = result.usage?.total_tokens ?? 0;

      return {
        content: text,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: { promptTokens, completionTokens, totalTokens },
        raw: result
      };
    });
  }

  async countTokens(messages: string | Message[]): Promise<number> {
    const text = typeof messages === 'string'
      ? messages
      : messages.map(msg => msg.content).join('\n');
    return Math.ceil(text.length / 4);
  }
}

/**
 * Maps the standard nanio message format to Claude Messages API content/role structure.
 */
function mapMessagesToClaude(messages: Message[]): { claudeMessages: any[]; systemPrompt?: string } {
  let systemPrompt: string | undefined;

  // Filter out system messages and merge them into a single prompt
  const filtered = messages.filter(msg => {
    if (msg.role === 'system') {
      systemPrompt = (systemPrompt ? systemPrompt + '\n' : '') + msg.content;
      return false;
    }
    return true;
  });

  const claudeMessages = filtered.map(msg => {
    let role: string = msg.role;
    if (role === 'tool') {
      role = 'user'; // Tool results are submitted under user role in Anthropic
    }

    if (msg.toolCalls && msg.toolCalls.length > 0) {
      return {
        role: 'assistant',
        content: [
          { type: 'text', text: msg.content || '' },
          ...msg.toolCalls.map(tc => ({
            type: 'tool_use',
            id: tc.id,
            name: tc.name,
            input: tc.arguments
          }))
        ]
      };
    }

    if (msg.role === 'tool') {
      return {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: msg.toolCallId || '',
            content: msg.content
          }
        ]
      };
    }

    return {
      role,
      content: msg.content
    };
  });

  return { claudeMessages, systemPrompt };
}

/**
 * Adapter for Anthropic Claude models using native fetch and resilient BaseAIClient.
 */
export class ClaudeModel extends BaseAIClient {
  public name: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'claude-3-5-sonnet-20241022',
    options?: { apiKey?: string; baseUrl?: string; uiPublisher?: any; perfTracker?: any }
  ) {
    super(modelName, options?.uiPublisher, options?.perfTracker);
    this.name = modelName;
    this.apiKey = options?.apiKey || process.env.ANTHROPIC_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://api.anthropic.com/v1';
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    if (!this.apiKey) {
      throw new Error(`Anthropic API key is missing. Set ANTHROPIC_API_KEY env variable or pass it directly.`);
    }

    return this.withRetry(async () => {
      const url = `${this.baseUrl}/messages`;
      const { claudeMessages, systemPrompt } = mapMessagesToClaude(messages);

      const body: any = {
        model: this.name,
        messages: claudeMessages,
        max_tokens: options?.maxTokens ?? 4096,
        temperature: options?.temperature ?? 0.7
      };

      if (systemPrompt) {
        body.system = systemPrompt;
      }

      if (options?.stopSequences && options.stopSequences.length > 0) {
        body.stop_sequences = options.stopSequences;
      }

      if (options?.tools && options.tools.length > 0) {
        body.tools = options.tools.map(tool => ({
          name: tool.name,
          description: tool.description,
          input_schema: tool.parameters
        }));
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errObj = new Error(`Anthropic API call failed with status ${response.status}: ${errorText}`);
        (errObj as any).status = response.status;
        throw errObj;
      }

      const result = await response.json();
      const contentBlocks = result.content || [];
      
      let text = '';
      const toolCalls: any[] = [];

      for (const block of contentBlocks) {
        if (block.type === 'text') {
          text += block.text;
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            name: block.name,
            arguments: block.input || {}
          });
        }
      }

      const promptTokens = result.usage?.input_tokens ?? 0;
      const completionTokens = result.usage?.output_tokens ?? 0;
      const totalTokens = promptTokens + completionTokens;

      return {
        content: text,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: { promptTokens, completionTokens, totalTokens },
        raw: result
      };
    });
  }

  async countTokens(messages: string | Message[]): Promise<number> {
    const text = typeof messages === 'string'
      ? messages
      : messages.map(msg => msg.content).join('\n');
    return Math.ceil(text.length / 4);
  }
}

/**
 * Adapter for xAI Grok models using native fetch and resilient BaseAIClient.
 */
export class XAIModel extends BaseAIClient {
  public name: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'grok-2',
    options?: { apiKey?: string; baseUrl?: string; uiPublisher?: any; perfTracker?: any }
  ) {
    super(modelName, options?.uiPublisher, options?.perfTracker);
    this.name = modelName;
    this.apiKey = options?.apiKey || process.env.XAI_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://api.x.ai/v1';
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    if (!this.apiKey) {
      throw new Error(`xAI API key is missing. Set XAI_API_KEY env variable or pass it directly.`);
    }

    return this.withRetry(async () => {
      const url = `${this.baseUrl}/chat/completions`;

      const xaiMessages = messages.map(msg => {
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          return {
            role: 'assistant',
            content: msg.content || null,
            tool_calls: msg.toolCalls.map(tc => ({
              id: tc.id,
              type: 'function',
              function: {
                name: tc.name,
                arguments: JSON.stringify(tc.arguments)
              }
            }))
          };
        }

        if (msg.role === 'tool') {
          return {
            role: 'tool',
            tool_call_id: msg.toolCallId,
            content: msg.content
          };
        }

        return {
          role: msg.role,
          content: msg.content
        };
      });

      const body: any = {
        model: this.name,
        messages: xaiMessages,
        temperature: options?.temperature ?? 0.7,
        max_tokens: options?.maxTokens,
        stop: options?.stopSequences
      };

      if (options?.tools && options.tools.length > 0) {
        body.tools = options.tools.map(tool => ({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters
          }
        }));
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errObj = new Error(`xAI API call failed with status ${response.status}: ${errorText}`);
        (errObj as any).status = response.status;
        throw errObj;
      }

      const result = await response.json();
      const choice = result.choices?.[0];
      if (!choice) {
        throw new Error(`xAI returned an empty response. Raw result: ${JSON.stringify(result)}`);
      }

      const text = choice.message?.content || '';
      const toolCalls: ToolCall[] = [];

      if (choice.message?.tool_calls) {
        for (const tc of choice.message.tool_calls) {
          if (tc.type === 'function') {
            let args: Record<string, any> = {};
            try {
              args = JSON.parse(tc.function.arguments || '{}');
            } catch {
              args = { _raw: tc.function.arguments };
            }
            toolCalls.push({
              id: tc.id,
              name: tc.function.name,
              arguments: args
            });
          }
        }
      }

      const promptTokens = result.usage?.prompt_tokens ?? 0;
      const completionTokens = result.usage?.completion_tokens ?? 0;
      const totalTokens = result.usage?.total_tokens ?? 0;

      return {
        content: text,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        usage: { promptTokens, completionTokens, totalTokens },
        raw: result
      };
    });
  }

  async countTokens(messages: string | Message[]): Promise<number> {
    const text = typeof messages === 'string'
      ? messages
      : messages.map(msg => msg.content).join('\n');
    return Math.ceil(text.length / 4);
  }
}

/**
 * Adapter for xAI image generation models (e.g. grok-imagine-image) using native fetch and resilient BaseAIClient.
 */
export class XAIImageClient extends BaseAIClient {
  public name: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'grok-imagine-image',
    options?: { apiKey?: string; baseUrl?: string; uiPublisher?: any; perfTracker?: any }
  ) {
    super(modelName, options?.uiPublisher, options?.perfTracker);
    this.name = modelName;
    this.apiKey = options?.apiKey || process.env.XAI_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://api.x.ai/v1';
  }

  async generate(messages: Message[], options?: GenerateOptions): Promise<LLMResponse> {
    if (!this.apiKey) {
      throw new Error(`xAI API key is missing. Set XAI_API_KEY env variable or pass it directly.`);
    }

    const prompt = messages.length > 0 ? messages[messages.length - 1].content : '';

    return this.withRetry(async () => {
      const url = `${this.baseUrl}/images/generations`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: this.name,
          prompt: prompt,
          n: 1,
          response_format: 'url'
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        const errObj = new Error(`xAI image generation failed with status ${response.status}: ${errorText}`);
        (errObj as any).status = response.status;
        throw errObj;
      }

      const result = await response.json();
      const imageUrl = result.data?.[0]?.url || '';

      return {
        content: imageUrl,
        raw: result
      };
    });
  }

  async countTokens(messages: string | Message[]): Promise<number> {
    return 0;
  }
}
