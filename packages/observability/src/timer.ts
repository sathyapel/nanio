/**
 * Simple stopwatch utility to measure execution durations in milliseconds.
 */
export class Timer {
  private start = performance.now();

  get elapsedMs(): number {
    return performance.now() - this.start;
  }

  reset(): void {
    this.start = performance.now();
  }
}

/**
 * Execute an async operation and measure its latency duration.
 */
export async function timeOperation<T>(callback: () => Promise<T>): Promise<{ result: T; elapsedMs: number }> {
  const start = performance.now();
  const result = await callback();
  const elapsedMs = performance.now() - start;
  return { result, elapsedMs };
}
