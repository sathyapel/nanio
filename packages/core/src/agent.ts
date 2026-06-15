import { BaseModel } from './model.js';
import { BaseTool } from '@nanio/tools';
import { BaseMemory, BufferMemory } from './memory.js';
import { nanioEvents } from './events.js';
import { Message, LLMResponse } from './types.js';

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

/**
 * Base abstract Agent class. All agents inherit from this.
 */
export abstract class BaseAgent {
  public name: string;

  constructor(
    name: string,
    protected model: BaseModel,
    protected tools: BaseTool<any, any>[] = [],
    protected memory: BaseMemory = new BufferMemory()
  ) {
    this.name = name;
  }

  abstract run(prompt: string, metadata?: Record<string, any>): Promise<string>;
}

/**
 * A simple agent that directly forwards questions to the model without executing tools.
 */
export class SimpleAgent extends BaseAgent {
  async run(prompt: string, metadata?: Record<string, any>): Promise<string> {
    const runId = generateId();
    nanioEvents.emitAgentStart(runId, this.name, prompt);

    try {
      await this.memory.addMessage({ role: 'user', content: prompt });
      const messages = await this.memory.getMessages();

      nanioEvents.emitLLMStart(runId, this.model.name, messages);
      const response = await this.model.generate(messages);
      nanioEvents.emitLLMEnd(runId, this.model.name, response);

      await this.memory.addMessage({ role: 'assistant', content: response.content });
      nanioEvents.emitAgentEnd(runId, this.name, response.content);

      return response.content;
    } catch (error: any) {
      nanioEvents.emitAgentError(runId, this.name, error);
      throw error;
    }
  }
}

/**
 * An advanced agent executing a reasoning-and-action tool calling loop.
 */
export class ToolCallingAgent extends BaseAgent {
  async run(prompt: string, metadata?: Record<string, any>): Promise<string> {
    const runId = generateId();
    nanioEvents.emitAgentStart(runId, this.name, prompt);

    try {
      await this.memory.addMessage({ role: 'user', content: prompt });

      let iterations = 0;
      const maxIterations = 8;

      while (iterations < maxIterations) {
        iterations++;

        const messages = await this.memory.getMessages();
        const toolSchemas = this.tools.map(t => t.toJSONSchema());

        nanioEvents.emitLLMStart(runId, this.model.name, messages);
        let response: LLMResponse;
        try {
          response = await this.model.generate(messages, { tools: toolSchemas.length > 0 ? toolSchemas : undefined });
          nanioEvents.emitLLMEnd(runId, this.model.name, response);
        } catch (llmError: any) {
          nanioEvents.emitLLMError(runId, this.model.name, llmError);
          throw llmError;
        }

        const assistantMessage: Message = {
          role: 'assistant',
          content: response.content,
          toolCalls: response.toolCalls,
        };
        await this.memory.addMessage(assistantMessage);

        // Terminate loop if there are no tool execution requests from the model
        if (!response.toolCalls || response.toolCalls.length === 0) {
          nanioEvents.emitAgentEnd(runId, this.name, response.content);
          return response.content;
        }

        // Execute tool calls sequentially
        for (const toolCall of response.toolCalls) {
          const tool = this.tools.find(t => t.name === toolCall.name);
          if (!tool) {
            const errorMsg = `Tool ${toolCall.name} not found in agent configuration.`;
            nanioEvents.emitToolError(runId, toolCall.name, new Error(errorMsg));
            await this.memory.addMessage({
              role: 'tool',
              content: `Error: Tool '${toolCall.name}' was not found.`,
              toolCallId: toolCall.id,
              name: toolCall.name
            });
            continue;
          }

          nanioEvents.emitToolStart(runId, tool.name, toolCall.arguments);
          try {
            const result = await tool.run(toolCall.arguments, metadata);
            const stringResult = typeof result === 'string' ? result : JSON.stringify(result);
            nanioEvents.emitToolEnd(runId, tool.name, result);

            await this.memory.addMessage({
              role: 'tool',
              content: stringResult,
              toolCallId: toolCall.id,
              name: tool.name
            });
          } catch (toolExecError: any) {
            nanioEvents.emitToolError(runId, tool.name, toolExecError);
            await this.memory.addMessage({
              role: 'tool',
              content: `Error: ${toolExecError.message || toolExecError}`,
              toolCallId: toolCall.id,
              name: tool.name
            });
          }
        }
      }

      throw new Error(`Agent loop reached limit of ${maxIterations} iterations without finishing.`);
    } catch (error: any) {
      nanioEvents.emitAgentError(runId, this.name, error);
      throw error;
    }
  }
}
