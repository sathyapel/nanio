import {
  nanioEvents,
  AgentStartEvent,
  AgentEndEvent,
  AgentErrorEvent,
  LLMStartEvent,
  LLMEndEvent,
  LLMErrorEvent,
  ToolStartEvent,
  ToolEndEvent,
  ToolErrorEvent
} from '@nanio/core';

/**
 * Base abstract class for Tracing and Telemetry exporters.
 */
export abstract class BaseTracer {
  abstract init(): void;
  abstract destroy(): void;
}

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  red: '\x1b[31m',
};

/**
 * A beautiful Console-based telemetry tracer that registers event hooks and logs operations in real-time.
 */
export class ConsoleTracer extends BaseTracer {
  private startTimeMap = new Map<string, number>();

  init() {
    nanioEvents.on('agent:start', this.handleAgentStart);
    nanioEvents.on('agent:end', this.handleAgentEnd);
    nanioEvents.on('agent:error', this.handleAgentError);

    nanioEvents.on('llm:start', this.handleLLMStart);
    nanioEvents.on('llm:end', this.handleLLMEnd);
    nanioEvents.on('llm:error', this.handleLLMError);

    nanioEvents.on('tool:start', this.handleToolStart);
    nanioEvents.on('tool:end', this.handleToolEnd);
    nanioEvents.on('tool:error', this.handleToolError);
  }

  destroy() {
    nanioEvents.off('agent:start', this.handleAgentStart);
    nanioEvents.off('agent:end', this.handleAgentEnd);
    nanioEvents.off('agent:error', this.handleAgentError);

    nanioEvents.off('llm:start', this.handleLLMStart);
    nanioEvents.off('llm:end', this.handleLLMEnd);
    nanioEvents.off('llm:error', this.handleLLMError);

    nanioEvents.off('tool:start', this.handleToolStart);
    nanioEvents.off('tool:end', this.handleToolEnd);
    nanioEvents.off('tool:error', this.handleToolError);
  }

  private handleAgentStart = (event: AgentStartEvent) => {
    this.startTimeMap.set(event.runId, event.timestamp);
    console.log(`\n${colors.magenta}${colors.bright}[Agent Start]${colors.reset} Name: ${event.agentName} | Run ID: ${event.runId}`);
    console.log(`${colors.dim}Prompt: "${event.prompt}"${colors.reset}`);
  };

  private handleAgentEnd = (event: AgentEndEvent) => {
    const start = this.startTimeMap.get(event.runId) || event.timestamp;
    const duration = event.timestamp - start;
    console.log(`${colors.magenta}${colors.bright}[Agent Success]${colors.reset} Name: ${event.agentName} | Run ID: ${event.runId} | Duration: ${duration}ms`);
    console.log(`${colors.green}Final Output:${colors.reset} ${event.output}`);
  };

  private handleAgentError = (event: AgentErrorEvent) => {
    const start = this.startTimeMap.get(event.runId) || event.timestamp;
    const duration = event.timestamp - start;
    console.log(`${colors.red}${colors.bright}[Agent Error]${colors.reset} Name: ${event.agentName} | Run ID: ${event.runId} | Duration: ${duration}ms`);
    console.error(`${colors.red}Error detail:${colors.reset}`, event.error);
  };

  private handleLLMStart = (event: LLMStartEvent) => {
    const key = `${event.runId}:llm`;
    this.startTimeMap.set(key, event.timestamp);
    console.log(`  ${colors.blue}[LLM Request]${colors.reset} Model: ${event.modelName}`);
  };

  private handleLLMEnd = (event: LLMEndEvent) => {
    const key = `${event.runId}:llm`;
    const start = this.startTimeMap.get(key) || event.timestamp;
    const duration = event.timestamp - start;
    const usage = event.response.usage
      ? `(${event.response.usage.promptTokens} prompt, ${event.response.usage.completionTokens} completion, ${event.response.usage.totalTokens} total tokens)`
      : '';
    console.log(`  ${colors.blue}[LLM Success]${colors.reset} Model: ${event.modelName} | Duration: ${duration}ms | Tokens: ${usage}`);
    if (event.response.content) {
      console.log(`    ${colors.dim}Content: "${event.response.content.trim().slice(0, 150)}${event.response.content.length > 150 ? '...' : ''}"${colors.reset}`);
    }
    if (event.response.toolCalls) {
      console.log(`    ${colors.cyan}Tool Calls Requested:${colors.reset} ${JSON.stringify(event.response.toolCalls)}`);
    }
  };

  private handleLLMError = (event: LLMErrorEvent) => {
    const key = `${event.runId}:llm`;
    const start = this.startTimeMap.get(key) || event.timestamp;
    const duration = event.timestamp - start;
    console.log(`  ${colors.red}[LLM Error]${colors.reset} Model: ${event.modelName} | Duration: ${duration}ms`);
    console.error(`    ${colors.red}Error:${colors.reset}`, event.error.message || event.error);
  };

  private handleToolStart = (event: ToolStartEvent) => {
    const key = `${event.runId}:tool:${event.toolName}`;
    this.startTimeMap.set(key, event.timestamp);
    console.log(`    ${colors.green}[Tool Call]${colors.reset} Tool: ${event.toolName}`);
    console.log(`      ${colors.dim}Arguments: ${JSON.stringify(event.input)}${colors.reset}`);
  };

  private handleToolEnd = (event: ToolEndEvent) => {
    const key = `${event.runId}:tool:${event.toolName}`;
    const start = this.startTimeMap.get(key) || event.timestamp;
    const duration = event.timestamp - start;
    console.log(`    ${colors.green}[Tool Success]${colors.reset} Tool: ${event.toolName} | Duration: ${duration}ms`);
    console.log(`      ${colors.dim}Result: ${JSON.stringify(event.output)}${colors.reset}`);
  };

  private handleToolError = (event: ToolErrorEvent) => {
    const key = `${event.runId}:tool:${event.toolName}`;
    const start = this.startTimeMap.get(key) || event.timestamp;
    const duration = event.timestamp - start;
    console.log(`    ${colors.red}[Tool Error]${colors.reset} Tool: ${event.toolName} | Duration: ${duration}ms`);
    console.error(`      ${colors.red}Error:${colors.reset}`, event.error.message || event.error);
  };
}
