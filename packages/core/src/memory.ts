import { Message } from './types.js';

/**
 * Base abstract Memory class. Custom memory systems (e.g. database, redis, window-based) inherit from this.
 */
export abstract class BaseMemory {
  abstract addMessage(message: Message): Promise<void>;
  abstract getMessages(): Promise<Message[]>;
  abstract clear(): Promise<void>;
}

/**
 * Simple concrete memory implementation storing messages in memory.
 */
export class BufferMemory extends BaseMemory {
  protected messages: Message[] = [];

  constructor(initialMessages: Message[] = []) {
    super();
    this.messages = [...initialMessages];
  }

  async addMessage(message: Message): Promise<void> {
    this.messages.push(message);
  }

  async getMessages(): Promise<Message[]> {
    return [...this.messages];
  }

  async clear(): Promise<void> {
    this.messages = [];
  }
}
