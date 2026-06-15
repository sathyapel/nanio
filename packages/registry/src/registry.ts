import { BaseModel } from '@nanio/core';

/**
 * Registry to manage available BaseModel instances in the application.
 */
export class ModelRegistry {
  private models = new Map<string, BaseModel>();

  register(name: string, model: BaseModel): void {
    this.models.set(name, model);
  }

  get(name: string): BaseModel | undefined {
    return this.models.get(name);
  }

  list(): string[] {
    return Array.from(this.models.keys());
  }

  clear(): void {
    this.models.clear();
  }
}
