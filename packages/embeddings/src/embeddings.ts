/**
 * Base abstract class for generating embeddings.
 */
export abstract class BaseEmbeddings {
  abstract embedQuery(text: string): Promise<number[]>;
  abstract embedDocuments(texts: string[]): Promise<number[][]>;
}

/**
 * Concrete implementation for Gemini Embeddings using native fetch.
 */
export class GeminiEmbeddings extends BaseEmbeddings {
  public model: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'gemini-embedding-001',
    options?: { apiKey?: string; baseUrl?: string }
  ) {
    super();
    this.model = modelName;
    this.apiKey = options?.apiKey || process.env.GEMINI_API_KEY || '';
    this.baseUrl = options?.baseUrl || 'https://generativelanguage.googleapis.com';
  }

  async embedQuery(text: string): Promise<number[]> {
    if (!this.apiKey) {
      throw new Error(`Gemini API key is missing. Set GEMINI_API_KEY env variable or pass it to GeminiEmbeddings.`);
    }

    const url = `${this.baseUrl}/v1/models/${this.model}:embedContent?key=${this.apiKey}`;
    const body = {
      content: {
        parts: [{ text }]
      }
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Gemini Embeddings call failed with status ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    const values = result.embedding?.values;
    if (!values) {
      throw new Error(`Gemini Embeddings response did not contain embedding values: ${JSON.stringify(result)}`);
    }

    return values;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error(`Gemini API key is missing. Set GEMINI_API_KEY env variable or pass it to GeminiEmbeddings.`);
    }

    if (texts.length === 0) return [];

    const CHUNK_SIZE = 100;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += CHUNK_SIZE) {
      const chunk = texts.slice(i, i + CHUNK_SIZE);
      const url = `${this.baseUrl}/v1/models/${this.model}:batchEmbedContents?key=${this.apiKey}`;
      const body = {
        requests: chunk.map(text => ({
          model: `models/${this.model}`,
          content: {
            parts: [{ text }]
          }
        }))
      };

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Gemini batchEmbedContents call failed with status ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      const embeddings = result.embeddings;
      if (!embeddings || embeddings.length !== chunk.length) {
        throw new Error(`Gemini batchEmbedContents response was invalid: ${JSON.stringify(result)}`);
      }

      for (const emb of embeddings) {
        if (!emb.values) {
          throw new Error(`Gemini batchEmbedContents response element did not contain values`);
        }
        allEmbeddings.push(emb.values);
      }
    }

    return allEmbeddings;
  }
}
