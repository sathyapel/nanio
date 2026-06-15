import { EventEmitter } from 'events';
import { Message, LLMResponse } from './types.js';

export interface AgentStartEvent {
  runId: string;
  agentName: string;
  prompt: string;
  timestamp: number;
}

export interface AgentEndEvent {
  runId: string;
  agentName: string;
  output: string;
  timestamp: number;
}

export interface AgentErrorEvent {
  runId: string;
  agentName: string;
  error: Error;
  timestamp: number;
}

export interface LLMStartEvent {
  runId: string;
  modelName: string;
  messages: Message[];
  timestamp: number;
}

export interface LLMEndEvent {
  runId: string;
  modelName: string;
  response: LLMResponse;
  timestamp: number;
}

export interface LLMErrorEvent {
  runId: string;
  modelName: string;
  error: Error;
  timestamp: number;
}

export interface ToolStartEvent {
  runId: string;
  toolName: string;
  input: any;
  timestamp: number;
}

export interface ToolEndEvent {
  runId: string;
  toolName: string;
  output: any;
  timestamp: number;
}

export interface ToolErrorEvent {
  runId: string;
  toolName: string;
  error: Error;
  timestamp: number;
}

export class NanioEventBus extends EventEmitter {
  emitAgentStart(runId: string, agentName: string, prompt: string) {
    const data: AgentStartEvent = { runId, agentName, prompt, timestamp: Date.now() };
    this.emit('agent:start', data);
  }
  emitAgentEnd(runId: string, agentName: string, output: string) {
    const data: AgentEndEvent = { runId, agentName, output, timestamp: Date.now() };
    this.emit('agent:end', data);
  }
  emitAgentError(runId: string, agentName: string, error: Error) {
    const data: AgentErrorEvent = { runId, agentName, error, timestamp: Date.now() };
    this.emit('agent:error', data);
  }

  emitLLMStart(runId: string, modelName: string, messages: Message[]) {
    const data: LLMStartEvent = { runId, modelName, messages, timestamp: Date.now() };
    this.emit('llm:start', data);
  }
  emitLLMEnd(runId: string, modelName: string, response: LLMResponse) {
    const data: LLMEndEvent = { runId, modelName, response, timestamp: Date.now() };
    this.emit('llm:end', data);
  }
  emitLLMError(runId: string, modelName: string, error: Error) {
    const data: LLMErrorEvent = { runId, modelName, error, timestamp: Date.now() };
    this.emit('llm:error', data);
  }

  emitToolStart(runId: string, toolName: string, input: any) {
    const data: ToolStartEvent = { runId, toolName, input, timestamp: Date.now() };
    this.emit('tool:start', data);
  }
  emitToolEnd(runId: string, toolName: string, output: any) {
    const data: ToolEndEvent = { runId, toolName, output, timestamp: Date.now() };
    this.emit('tool:end', data);
  }
  emitToolError(runId: string, toolName: string, error: Error) {
    const data: ToolErrorEvent = { runId, toolName, error, timestamp: Date.now() };
    this.emit('tool:error', data);
  }
}

export const nanioEvents = new NanioEventBus();
