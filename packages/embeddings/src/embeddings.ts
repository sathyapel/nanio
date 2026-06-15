import { pipeline } from '@huggingface/transformers';

/**
 * Base abstract class for generating embeddings.
 * All providers (GeminiEmbeddings, OpenAIEmbeddings, TransformersEmbeddings)
 * extend this class, making them interchangeable in IHR, ZVecVectorStore,
 * and any other component that accepts a BaseEmbeddings instance.
 */
export abstract class BaseEmbeddings {
  /**
   * Human-readable provider name, used for observability logging.
   * Subclasses should set `public model` so this reflects the active model name.
   */
  abstract embedQuery(text: string): Promise<number[]>;
  abstract embedDocuments(texts: string[]): Promise<number[][]>;
}

/**
 * Cloud provider: Google Gemini Embeddings.
 * Default model: gemini-embedding-001 → 768-dimensional vectors.
 * Requires GEMINI_API_KEY environment variable or apiKey option.
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

/**
 * Cloud provider: OpenAI Embeddings.
 * Default model: text-embedding-3-small → 1536-dimensional vectors.
 * Also supports text-embedding-3-large (3072-dim) and ada-002 (1536-dim).
 * Requires OPENAI_API_KEY environment variable or apiKey option.
 */
export class OpenAIEmbeddings extends BaseEmbeddings {
  public model: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(
    modelName: string = 'text-embedding-3-small',
    options?: { apiKey?: string; baseUrl?: string }
  ) {
    super();
    this.model = modelName;
    this.apiKey = options?.apiKey || process.env.OPENAI_API_KEY || '';
    this.baseUrl = (options?.baseUrl || 'https://api.openai.com').replace(/\/$/, '');
  }

  async embedQuery(text: string): Promise<number[]> {
    const results = await this.embedDocuments([text]);
    return results[0];
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error(`OpenAI API key is missing. Set OPENAI_API_KEY env variable or pass it to OpenAIEmbeddings.`);
    }
    if (texts.length === 0) return [];

    // OpenAI accepts up to 2048 items per batch request
    const CHUNK_SIZE = 2048;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += CHUNK_SIZE) {
      const chunk = texts.slice(i, i + CHUNK_SIZE);
      const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({ model: this.model, input: chunk })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OpenAI Embeddings call failed with status ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      if (!result.data || result.data.length !== chunk.length) {
        throw new Error(`OpenAI Embeddings response was invalid: ${JSON.stringify(result)}`);
      }

      // Sort by index to guarantee insertion order is preserved
      result.data.sort((a: any, b: any) => a.index - b.index);
      for (const item of result.data) {
        allEmbeddings.push(item.embedding);
      }
    }

    return allEmbeddings;
  }
}

/**
 * Local provider: SentenceTransformer Embeddings via Hugging Face Transformers.js.
 * Runs fully offline using ONNX Runtime — no API key required.
 * Default model: Xenova/all-MiniLM-L6-v2 → 384-dimensional normalized vectors.
 * Also compatible with any Xenova/* sentence-transformer model on Hugging Face Hub.
 */
export class TransformersEmbeddings extends BaseEmbeddings {
  private extractorPromise: any = null;
  public model: string;

  constructor(modelName: string = 'Xenova/all-MiniLM-L6-v2') {
    super();
    this.model = modelName;
  }

  private async getExtractor() {
    if (!this.extractorPromise) {
      this.extractorPromise = pipeline('feature-extraction', this.model);
    }
    return this.extractorPromise;
  }

  async embedQuery(text: string): Promise<number[]> {
    const extractor = await this.getExtractor();
    const output = await extractor(text, {
      pooling: 'mean',
      normalize: true,
    });
    return Array.from(output.data);
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      const emb = await this.embedQuery(text);
      results.push(emb);
    }
    return results;
  }
}
