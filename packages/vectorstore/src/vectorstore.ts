import { BaseEmbeddings } from '@nanio/embeddings';
import pg from 'pg';
import crypto from 'crypto';

export interface Document {
  pageContent: string;
  metadata?: Record<string, any>;
}

interface VectorItem {
  document: Document;
  vector: number[];
}

/**
 * Base abstract class for Vector Store implementations.
 */
export abstract class BaseVectorStore {
  constructor(protected embeddings: BaseEmbeddings) {}

  abstract addDocuments(documents: Document[]): Promise<void>;
  abstract similaritySearch(query: string, k?: number): Promise<Document[]>;
  abstract similaritySearchWithScore(query: string, k?: number): Promise<[Document, number][]>;
}

function dotProduct(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

function magnitude(a: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = dotProduct(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

/**
 * An in-memory vector store that computes similarity using Cosine Similarity.
 */
export class MemoryVectorStore extends BaseVectorStore {
  private items: VectorItem[] = [];

  async addDocuments(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    // Generate embeddings for all documents in a batch call
    const texts = documents.map(doc => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);

    for (let i = 0; i < documents.length; i++) {
      this.items.push({
        document: documents[i],
        vector: vectors[i]
      });
    }
  }

  async similaritySearch(query: string, k: number = 4): Promise<Document[]> {
    const results = await this.similaritySearchWithScore(query, k);
    return results.map(([doc]) => doc);
  }

  async similaritySearchWithScore(query: string, k: number = 4): Promise<[Document, number][]> {
    if (this.items.length === 0) return [];

    // Embed the query
    const queryVector = await this.embeddings.embedQuery(query);

    // Calculate similarity scores for all indexed items
    const scoredItems: [Document, number][] = this.items.map(item => {
      const score = cosineSimilarity(queryVector, item.vector);
      return [item.document, score];
    });

    // Sort descending by score and return top k
    return scoredItems
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }
}

/**
 * A MongoDB-backed persistent vector store.
 */
export class MongoVectorStore extends BaseVectorStore {
  constructor(embeddings: BaseEmbeddings, private collection: any) {
    super(embeddings);
  }

  async addDocuments(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    const texts = documents.map(doc => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);

    const docsToInsert = documents.map((doc, idx) => ({
      pageContent: doc.pageContent,
      metadata: doc.metadata || {},
      vector: vectors[idx]
    }));

    await this.collection.insertMany(docsToInsert);
  }

  async similaritySearch(query: string, k: number = 4): Promise<Document[]> {
    const results = await this.similaritySearchWithScore(query, k);
    return results.map(([doc]) => doc);
  }

  async similaritySearchWithScore(query: string, k: number = 4): Promise<[Document, number][]> {
    const queryVector = await this.embeddings.embedQuery(query);

    const docs = await this.collection.find({}).toArray();
    if (docs.length === 0) return [];

    const scoredItems: [Document, number][] = docs.map((doc: any) => {
      const score = cosineSimilarity(queryVector, doc.vector);
      const document: Document = {
        pageContent: doc.pageContent,
        metadata: doc.metadata
      };
      return [document, score];
    });

    return scoredItems
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }
}

/**
 * A PostgreSQL-backed vector store using pgvector.
 */
export class PgVectorStore extends BaseVectorStore {
  constructor(
    embeddings: BaseEmbeddings,
    private client: pg.Client | pg.Pool,
    private tableName: string = 'documents'
  ) {
    super(embeddings);
  }

  async initializeSchema(dimension: number = 768): Promise<void> {
    await this.client.query('CREATE EXTENSION IF NOT EXISTS vector;');
    await this.client.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB NOT NULL,
        embedding VECTOR(${dimension})
      );
    `);
  }

  async addDocuments(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    const texts = documents.map(doc => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);

    for (let i = 0; i < documents.length; i++) {
      const vectorStr = `[${vectors[i].join(',')}]`;
      await this.client.query(
        `INSERT INTO ${this.tableName} (content, metadata, embedding) VALUES ($1, $2, $3)`,
        [documents[i].pageContent, JSON.stringify(documents[i].metadata || {}), vectorStr]
      );
    }
  }

  async similaritySearch(query: string, k: number = 4): Promise<Document[]> {
    const results = await this.similaritySearchWithScore(query, k);
    return results.map(([doc]) => doc);
  }

  async similaritySearchWithScore(query: string, k: number = 4): Promise<[Document, number][]> {
    const queryVector = await this.embeddings.embedQuery(query);
    const vectorStr = `[${queryVector.join(',')}]`;

    const res = await this.client.query(
      `SELECT content, metadata, (embedding <=> $1) as distance FROM ${this.tableName} ORDER BY embedding <=> $1 LIMIT $2`,
      [vectorStr, k]
    );

    return res.rows.map(row => {
      const doc: Document = {
        pageContent: row.content,
        metadata: row.metadata
      };
      const similarity = 1.0 - parseFloat(row.distance);
      return [doc, similarity];
    });
  }
}

/**
 * A Qdrant-backed vector store using direct REST calls via fetch.
 */
export class QdrantVectorStore extends BaseVectorStore {
  private url: string;
  private apiKey?: string;
  private collectionName: string;

  constructor(
    embeddings: BaseEmbeddings,
    options: { url: string; apiKey?: string; collectionName?: string }
  ) {
    super(embeddings);
    this.url = options.url.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.collectionName = options.collectionName || 'documents';
  }

  private async request(path: string, options: RequestInit = {}): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    if (this.apiKey) {
      headers['api-key'] = this.apiKey;
    }
    const response = await fetch(`${this.url}${path}`, {
      ...options,
      headers: {
        ...headers,
        ...options.headers
      }
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Qdrant API error: ${response.status} - ${text}`);
    }

    return response.json();
  }

  async initializeCollection(dimension: number = 768): Promise<void> {
    try {
      // Check if collection exists
      await this.request(`/collections/${this.collectionName}`);
    } catch (err) {
      // Create collection if it doesn't exist
      await this.request(`/collections/${this.collectionName}`, {
        method: 'PUT',
        body: JSON.stringify({
          vectors: {
            size: dimension,
            distance: 'Cosine'
          }
        })
      });
    }
  }

  async addDocuments(documents: Document[]): Promise<void> {
    if (documents.length === 0) return;

    const texts = documents.map(doc => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);

    const points = documents.map((doc, idx) => ({
      id: crypto.randomUUID(),
      vector: vectors[idx],
      payload: {
        pageContent: doc.pageContent,
        metadata: doc.metadata || {}
      }
    }));

    await this.request(`/collections/${this.collectionName}/points`, {
      method: 'PUT',
      body: JSON.stringify({ points })
    });
  }

  async similaritySearch(query: string, k: number = 4): Promise<Document[]> {
    const results = await this.similaritySearchWithScore(query, k);
    return results.map(([doc]) => doc);
  }

  async similaritySearchWithScore(query: string, k: number = 4): Promise<[Document, number][]> {
    const queryVector = await this.embeddings.embedQuery(query);

    const res = await this.request(`/collections/${this.collectionName}/points/search`, {
      method: 'POST',
      body: JSON.stringify({
        vector: queryVector,
        limit: k,
        with_payload: true,
        with_vector: false
      })
    });

    return (res.result || []).map((hit: any) => {
      const doc: Document = {
        pageContent: hit.payload?.pageContent || '',
        metadata: hit.payload?.metadata || {}
      };
      return [doc, hit.score];
    });
  }
}

