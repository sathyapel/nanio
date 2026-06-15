import pg from 'pg';
import fs from 'fs';
import { BaseModel, Message, ChatConfig } from '@nanio/core';
import { BaseEmbeddings } from '@nanio/embeddings';
import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType
} from '@zvec/zvec';
import { getLogger, timeOperation } from '@nanio/observability';

const logger = getLogger('IndexHydratedRAG');

export interface IngestSection {
  section_id: string;
  parent_id: string | null;
  level: number;
  heading: string;
  content: string; // full text for leaf nodes
  summary: string; // 2-3 sentences summary + keywords
}

export interface RetrieveOptions {
  /** Semantic search top-N (default 5). */
  limit?: number;
  /** Number of sections to keep after TF-IDF re-rank (default 3). */
  tfidfLimit?: number;
  /**
   * Custom stopword set passed to TF-IDF tokeniser.
   * Always merged on top of the LLM-generated or DEFAULT stopwords.
   */
  stopwords?: Set<string>;
  /**
   * When true, calls the configured LLM to:
   *   1. Detect the primary language of the candidate corpus.
   *   2. Return a comprehensive, language-specific stopword list.
   * The LLM-generated words are merged with DEFAULT_STOPWORDS before TF-IDF runs.
   * This ensures correct stopword handling for any language (French, German, Spanish, etc.)
   * without requiring manual stopword lists per language.
   * Default: false (uses DEFAULT_STOPWORDS only).
   */
  autoStopwords?: boolean;
}

export interface RetrieveResponse {
  answer: string;
  context: string;
  retrievedSections: string[];
  /** Always 'tfidf' — TF-IDF re-rank runs unconditionally as per reference notebook. */
  path: 'tfidf';
}

/**
 * Computes cosine similarity between two vectors.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export const DEFAULT_STOPWORDS = new Set([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
  'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
  'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
  'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
  'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
]);

/**
 * Tokenizes text: lowercase → strip punctuation → split → remove stopwords.
 */
export function tokenize(text: string, stopwords: Set<string> = DEFAULT_STOPWORDS): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0 && !stopwords.has(t));
}

/**
 * Generates unigrams + bigrams from a token list.
 * Equivalent to sklearn TfidfVectorizer(ngram_range=(1, 2)).
 */
export function buildNgrams(tokens: string[]): string[] {
  const result: string[] = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    result.push(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return result;
}

/**
 * Automatically derives corpus-level stopwords from candidate texts.
 *
 * Any term appearing in >= `threshold` fraction of the candidate documents
 * is added to the stopword set (merged onto `baseStopwords`).
 * This mirrors sklearn's max_df behaviour.
 *
 * @param texts       - raw text strings for each candidate section
 * @param threshold   - document-frequency fraction cutoff (default 0.85)
 * @param baseStopwords - seed stopword set (DEFAULT_STOPWORDS)
 */
export function generateCorpusStopwords(
  texts: string[],
  threshold: number = 0.85,
  baseStopwords: Set<string> = DEFAULT_STOPWORDS
): Set<string> {
  const N = texts.length;
  if (N === 0) return new Set(baseStopwords);

  // Count document frequency WITHOUT removing stopwords first, so very common
  // function words in the corpus that somehow escaped the base list are caught.
  const df: Record<string, number> = {};
  for (const text of texts) {
    // Tokenize with an empty stopword set for df counting
    const terms = new Set(
      text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(t => t.length > 0)
    );
    for (const t of terms) df[t] = (df[t] || 0) + 1;
  }

  const combined = new Set(baseStopwords);
  for (const [term, count] of Object.entries(df)) {
    if (count / N >= threshold) combined.add(term);
  }
  return combined;
}

/**
 * TF-IDF cosine re-ranking — directly mirrors the reference notebook:
 *
 *   corpus      = [query] + section_documents
 *   vectorizer  = TfidfVectorizer(stop_words supplied, ngram_range=(1,2))
 *   tfidf_mat   = vectorizer.fit_transform(corpus)     ← IDF over full corpus
 *   scores      = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
 *
 * Differences from old tfidfScore:
 *   - IDF uses the **full** corpus (query + all section docs) — same as sklearn
 *   - Bigrams are included (ngram_range 1-2)
 *   - Final cosine similarity on L2-normalised TF-IDF vectors (not raw TF * IDF dot-product)
 *
 * @param query    - user query string
 * @param sections - candidate sections with { section_id, text }
 * @param stopwords - stopword set (DEFAULT_STOPWORDS or corpus-generated)
 */
export function tfidfCosineScore(
  query: string,
  sections: { section_id: string; text: string }[],
  stopwords: Set<string> = DEFAULT_STOPWORDS
): { section_id: string; score: number }[] {
  if (sections.length === 0) return [];

  // Build corpus: doc[0] = query, doc[1..n] = section texts
  const allTexts = [query, ...sections.map(s => s.text)];

  // Tokenize each document and expand to unigrams + bigrams
  const tokenized: string[][] = allTexts.map(t => buildNgrams(tokenize(t, stopwords)));

  // Build global vocabulary (insertion-order stable)
  const vocab = new Map<string, number>();
  for (const toks of tokenized) {
    for (const t of toks) {
      if (!vocab.has(t)) vocab.set(t, vocab.size);
    }
  }

  const V = vocab.size;
  const N = tokenized.length; // includes query

  // Document frequency: how many documents contain each term
  const df = new Float64Array(V);
  for (const toks of tokenized) {
    const seen = new Set(toks);
    for (const t of seen) {
      const idx = vocab.get(t);
      if (idx !== undefined) df[idx]++;
    }
  }

  // sklearn smooth IDF: log((1+N)/(1+df)) + 1
  const idf = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    idf[i] = Math.log((1 + N) / (1 + df[i])) + 1;
  }

  // Build TF-IDF vector and L2-normalise for each document
  const tfidfVecs: Float64Array[] = tokenized.map(toks => {
    const tf = new Float64Array(V);
    for (const t of toks) {
      const idx = vocab.get(t);
      if (idx !== undefined) tf[idx]++;
    }
    const total = toks.length || 1;
    const vec = new Float64Array(V);
    let norm = 0;
    for (let i = 0; i < V; i++) {
      vec[i] = (tf[i] / total) * idf[i];
      norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < V; i++) vec[i] /= norm;
    }
    return vec;
  });

  // Cosine similarity: query (index 0) vs each section (index 1..n)
  const queryVec = tfidfVecs[0];
  const scored = sections.map((s, i) => {
    const docVec = tfidfVecs[i + 1];
    let score = 0;
    for (let j = 0; j < V; j++) score += queryVec[j] * docVec[j];
    return { section_id: s.section_id, score };
  });

  return scored.sort((a, b) => b.score - a.score);
}

/**
 * @deprecated Use tfidfCosineScore instead.
 * Kept for backwards-compatibility with external consumers.
 */
export function tfidfScore(
  query: string,
  sections: { section_id: string; text: string }[],
  stopwords: Set<string> = DEFAULT_STOPWORDS
): { section_id: string; score: number }[] {
  return tfidfCosineScore(query, sections, stopwords);
}

/**
 * Dedup rule: parent absorbs child if child.startsWith(parent + ".")
 */
export function pruneSubtrees(dbIds: string[]): string[] {
  const sorted = [...new Set(dbIds)].sort((a, b) => a.localeCompare(b));
  const pruned: string[] = [];
  for (const dbId of sorted) {
    const isChild = pruned.some(parentDbId => dbId.startsWith(parentDbId + '.'));
    if (!isChild) {
      pruned.push(dbId);
    }
  }
  return pruned;
}

export class IndexHydratedRAG {
  private dimension: number = 0;

  constructor(
    private db: pg.Client | pg.Pool,
    private embeddings: BaseEmbeddings,
    private model: BaseModel
  ) {}

  /**
   * Returns a human-readable name for the active embedding provider.
   * Uses the `model` property (present on GeminiEmbeddings, OpenAIEmbeddings,
   * TransformersEmbeddings) or falls back to the class constructor name.
   */
  get embeddingProviderName(): string {
    return (this.embeddings as any).model || this.embeddings.constructor.name;
  }

  /**
   * Initializes the database schema for IndexHydratedRAG.
   */
  async initializeSchema(): Promise<void> {
    await this.db.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT,
        source_url TEXT,
        ingested_at TIMESTAMPTZ DEFAULT now()
      );

      CREATE TABLE IF NOT EXISTS section_table (
        id SERIAL PRIMARY KEY,
        doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
        section_id TEXT,
        parent_id TEXT,
        level SMALLINT,
        heading TEXT,
        content TEXT,
        summary TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_section_table_doc_parent ON section_table (doc_id, parent_id);
    `);
  }

  /**
   * Helper to convert local section_id to database key.
   */
  private toDbId(docId: string, sectionId: string): string {
    return `${docId}__${sectionId}`;
  }

  /**
   * Helper to convert database key back to clean section_id.
   */
  private fromDbId(dbId: string): string {
    const idx = dbId.indexOf('__');
    if (idx !== -1) {
      return dbId.slice(idx + 2);
    }
    return dbId;
  }

  /**
   * Fetches the dimension of the embedding client dynamically.
   */
  private async getDimension(): Promise<number> {
    if (this.dimension === 0) {
      const dummyEmb = await this.embeddings.embedQuery("dummy");
      this.dimension = dummyEmb.length;
    }
    return this.dimension;
  }

  /**
   * Uses the configured LLM to:
   *   1. Detect the primary language of the candidate corpus.
   *   2. Return a comprehensive stopword list for that language.
   *
   * The model is prompted with a representative text sample (up to 1 200 chars)
   * and asked to respond in JSON:
   *
   *   { "language": "French", "languageCode": "fr", "stopwords": ["le", "la", ...] }
   *
   * The returned stopwords are merged with DEFAULT_STOPWORDS so the base
   * English set always applies as a fallback, even for multilingual corpora.
   *
   * Falls back to DEFAULT_STOPWORDS if the LLM response cannot be parsed.
   *
   * @param candidateTexts - raw text strings from the candidate sections
   */
  private async generateLLMStopwords(
    candidateTexts: string[]
  ): Promise<{ stopwords: Set<string>; language: string; languageCode: string }> {
    // Build a representative sample: concatenate unique texts, cap at 1 200 chars
    const sampleRaw = candidateTexts
      .map(t => t.trim())
      .filter(t => t.length > 0)
      .join(' ');
    const sample = sampleRaw.slice(0, 1200);

    const prompt = `You are a computational linguistics expert and a language-detection specialist.

Analyse the following text sample taken from a document retrieval corpus:

---
${sample}
---

Tasks:
1. Detect the primary language of the text.
2. Produce a comprehensive, high-quality stopword list for that language.
   Include: articles, prepositions, conjunctions, auxiliary verbs, pronouns,
   common adverbs, and any other high-frequency function words that carry
   no discriminative information for keyword search.
   Include BOTH lowercase and common contracted forms (e.g. "don't", "it's").

Return ONLY valid JSON with this exact schema — no prose, no markdown fences:
{
  "language": "<full language name, e.g. English>",
  "languageCode": "<ISO 639-1 code, e.g. en>",
  "stopwords": ["word1", "word2", ...]
}`;

    const { result: llmResponse, elapsedMs } = await timeOperation(
      async () => this.model.generate(
        [{ role: 'user', content: prompt }],
        { config: new ChatConfig({ jsonMode: true }) }
      )
    );

    let language = 'Unknown';
    let languageCode = 'xx';
    let llmWords: string[] = [];

    try {
      const parsed = JSON.parse(llmResponse.content);
      language     = parsed.language     || 'Unknown';
      languageCode = parsed.languageCode || 'xx';
      llmWords     = Array.isArray(parsed.stopwords) ? parsed.stopwords : [];
    } catch (err) {
      logger.warn('Failed to parse LLM stopword response; falling back to DEFAULT_STOPWORDS', {
        error: err instanceof Error ? err.message : String(err),
        rawSnippet: llmResponse.content.slice(0, 200)
      });
    }

    // Merge LLM words with DEFAULT_STOPWORDS (lowercase for consistency)
    const merged = new Set(DEFAULT_STOPWORDS);
    for (const w of llmWords) {
      const lower = w.toLowerCase().trim();
      if (lower.length > 0) merged.add(lower);
    }

    logger.info('LLM auto-stopwords generated', {
      language,
      languageCode,
      llmStopwordsCount: llmWords.length,
      totalStopwordsCount: merged.size,
      llmElapsedMs: elapsedMs
    });

    return { stopwords: merged, language, languageCode };
  }

  private async getZvecCollection(docId: string) {
    const collectionDir = `./zvec_data/${docId}`;
    if (fs.existsSync(collectionDir)) {
      return ZVecOpen(collectionDir);
    }
    const dim = await this.getDimension();
    const collectionSchema = new ZVecCollectionSchema({
      name: "ihr_sections",
      fields: [
        { name: "doc_id", dataType: ZVecDataType.STRING },
        { name: "section_id", dataType: ZVecDataType.STRING },
        { name: "summary", dataType: ZVecDataType.STRING },
        { name: "related_ids", dataType: ZVecDataType.STRING },
        { name: "sibling_ids", dataType: ZVecDataType.STRING }
      ],
      vectors: [
        {
          name: "embedding",
          dataType: ZVecDataType.VECTOR_FP32,
          dimension: dim,
          indexParams: { indexType: ZVecIndexType.HNSW, metricType: ZVecMetricType.COSINE }
        }
      ]
    });
    return ZVecCreateAndOpen(collectionDir, collectionSchema);
  }

  /**
   * Ingests a structured document tree into the relational and index tables.
   */
  async ingest(
    docId: string,
    title: string,
    url: string,
    sections: IngestSection[]
  ): Promise<void> {
    logger.info('Starting structured document tree ingestion', {
      docId,
      title,
      url,
      sectionsCount: sections.length,
      embeddingProvider: this.embeddingProviderName
    });

    // 1. Write document entry in PostgreSQL
    await this.db.query(
      `INSERT INTO documents (id, title, source_url) 
       VALUES ($1, $2, $3) 
       ON CONFLICT (id) DO UPDATE SET title = $2, source_url = $3`,
      [docId, title, url]
    );

    // 2. Clean previous sections in relational DB for this document to allow re-ingestion
    await this.db.query(`DELETE FROM section_table WHERE doc_id = $1`, [docId]);

    // 3. Write structured section outlines to section_table
    for (const sec of sections) {
      const dbSecId = this.toDbId(docId, sec.section_id);
      const dbParentId = sec.parent_id ? this.toDbId(docId, sec.parent_id) : null;
      await this.db.query(
        `INSERT INTO section_table (doc_id, section_id, parent_id, level, heading, content, summary) 
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [docId, dbSecId, dbParentId, sec.level, sec.heading, sec.content, sec.summary]
      );
    }

    // 4. One-time LLM map pass: Analyze outline to extract sibling_ids, related_ids, prerequisite_ids
    const outline = sections.map(s => ({
      section_id: s.section_id,
      parent_id: s.parent_id,
      heading: s.heading,
      summary: s.summary
    }));

    const outlinePrompt = `You are a document structure analyst.
For each section in the following document outline, analyze the context, headings, and summaries to identify:
1. related_ids — section_ids that this section cross-references (e.g. "see Appendix A" or related topics covered elsewhere).
2. sibling_ids — nearby section_ids in the same subtree or topic.
3. prerequisite_ids — section_ids that the reader should understand first before reading this section.

Here is the document outline:
${JSON.stringify(outline, null, 2)}

Return a JSON object where the keys are the section_id values, and the values are objects containing:
- related_ids: string[]
- sibling_ids: string[]
- prerequisite_ids: string[]

Example output format:
{
  "1.2.1": {
    "related_ids": ["2.3"],
    "sibling_ids": ["1.2.2"],
    "prerequisite_ids": ["1.2"]
  }
}`;

    const { result: mapResponse, elapsedMs: mapElapsedMs } = await timeOperation(
      async () => {
        return await this.model.generate([
          { role: 'user', content: outlinePrompt }
        ], { config: new ChatConfig({ jsonMode: true }) });
      }
    );

    let relationshipMap: Record<string, { related_ids?: string[], sibling_ids?: string[], prerequisite_ids?: string[] }> = {};
    try {
      relationshipMap = JSON.parse(mapResponse.content);
    } catch (err) {
      logger.warn('Could not parse outline structure LLM response. Defaulting relationships to empty.', { error: err instanceof Error ? err.message : String(err) });
    }

    // 5. Generate embeddings of enriched section text (Title + Heading + Body snippet)
    const richTexts = sections.map(s => {
      const bodySnippet = s.summary || s.content.slice(0, 200);
      return `${title} — ${s.heading}. ${bodySnippet}`;
    });
    
    const { result: embeddingsArray, elapsedMs: embedElapsedMs } = await timeOperation(
      async () => {
        return await this.embeddings.embedDocuments(richTexts);
      }
    );

    // 6. Setup local zvec collection folder, cleaning previous data first for idempotency
    const collectionDir = `./zvec_data/${docId}`;
    fs.rmSync(collectionDir, { recursive: true, force: true });
    const collection = await this.getZvecCollection(docId);
    try {
      // 7. Insert vectors + fields into Zvec collection
      for (let i = 0; i < sections.length; i++) {
        const sec = sections[i];
        const dbSecId = this.toDbId(docId, sec.section_id);
        const embedding = embeddingsArray[i];

        const relationships = relationshipMap[sec.section_id] || {};
        const relatedDbIds = (relationships.related_ids || []).map(id => this.toDbId(docId, id));
        const siblingDbIds = (relationships.sibling_ids || []).map(id => this.toDbId(docId, id));

        collection.insertSync({
          id: dbSecId,
          vectors: { "embedding": embedding },
          fields: {
            "doc_id": docId,
            "section_id": sec.section_id,
            "summary": sec.summary || '',
            "related_ids": JSON.stringify(relatedDbIds),
            "sibling_ids": JSON.stringify(siblingDbIds)
          }
        });
      }

      // 8. Optimize HNSW index
      collection.optimizeSync();
      logger.info('Structured document tree ingestion and indexing completed successfully', {
        docId,
        sectionsCount: sections.length,
        relationshipsFound: Object.keys(relationshipMap).length
      });
    } finally {
      collection.closeSync();
    }
  }

  private async performSimilaritySearch(docId: string, query: string, limit: number): Promise<string[]> {
    const { result: queryEmbedding } = await timeOperation(
      async () => {
        return await this.embeddings.embedQuery(query);
      }
    );
    const collection = await this.getZvecCollection(docId);
    try {
      const queryResult = await collection.query({
        fieldName: "embedding",
        vector: queryEmbedding,
        topk: limit
      });

      if (!Array.isArray(queryResult)) return [];
      return queryResult.map((res: any) => res.id);
    } finally {
      collection.closeSync();
    }
  }

  /**
   * Retrieves context and generates an answer using the zvec + relational IHR architecture.
   */
  async retrieve(
    query: string,
    docId: string,
    options?: RetrieveOptions
  ): Promise<RetrieveResponse> {
    logger.info('Starting tree-structure context retrieval (IHR)', {
      query,
      docId,
      limit: options?.limit,
      embeddingProvider: this.embeddingProviderName
    });
    const limit = options?.limit ?? 5;
    const tfidfLimit = options?.tfidfLimit ?? 3;

    // --- 3.1 LLM Call 1: Tool Agent ---
    const searchTool = {
      name: 'search_sections',
      description: 'Search for section IDs in the document index using semantic similarity. Returns matching section IDs.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query to match.'
          }
        },
        required: ['query']
      }
    };

    const systemPrompt = `You are a great tree search engineer.
Find which sections are relevant to the user query.
Use the search_sections tool to lookup candidates.
Return a JSON object containing:
{ "sections": ["section_id", ...] }`;

    const messages: Message[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query }
    ];

    let attempts = 0;
    const maxAttempts = 3;
    let { result: response } = await timeOperation(
      async () => {
        return await this.model.generate(messages, {
          tools: [searchTool],
          config: new ChatConfig({ jsonMode: true })
        });
      }
    );

    while (response.toolCalls && response.toolCalls.length > 0 && attempts < maxAttempts) {
      attempts++;
      messages.push({
        role: 'assistant',
        content: response.content,
        toolCalls: response.toolCalls
      });

      for (const tc of response.toolCalls) {
        if (tc.name === 'search_sections') {
          const tcQuery = tc.arguments.query || query;
          const matchedDbIds = await this.performSimilaritySearch(docId, tcQuery, limit);
          const cleanMatchedIds = matchedDbIds.map(dbId => this.fromDbId(dbId));
          messages.push({
            role: 'tool',
            name: tc.name,
            toolCallId: tc.id,
            content: JSON.stringify({ section_ids: cleanMatchedIds })
          });
        } else {
          messages.push({
            role: 'tool',
            name: tc.name,
            toolCallId: tc.id,
            content: `Error: Unknown tool ${tc.name}`
          });
        }
      }

      const agentRes = await timeOperation(
        async () => {
          return await this.model.generate(messages, {
            tools: [searchTool],
            config: new ChatConfig({ jsonMode: true })
          });
        }
      );
      response = agentRes.result;
    }

    let hitIds: string[] = [];
    try {
      const parsed = JSON.parse(response.content);
      if (parsed && Array.isArray(parsed.sections)) {
        hitIds = parsed.sections;
      } else if (parsed && Array.isArray(parsed)) {
        hitIds = parsed;
      }
    } catch {
      const matches = response.content.match(/\b\d+(\.\d+)*\b/g);
      if (matches) {
        hitIds = matches;
      }
    }

    let hitDbIds = hitIds.map(id => this.toDbId(docId, id));

    // Fallback: If agent fails, call similarity search directly
    if (hitDbIds.length === 0) {
      hitDbIds = await this.performSimilaritySearch(docId, query, limit);
    }

    // --- Expand candidates using pre-computed zvec related_ids & sibling_ids ---
    const collection = await this.getZvecCollection(docId);
    const expandedDbIds = new Set<string>(hitDbIds);

    try {
      for (const dbId of hitDbIds) {
        try {
          const fetched = collection.fetchSync(dbId);
          const doc = fetched[dbId];
          if (doc && doc.fields) {
            const related = doc.fields.related_ids ? JSON.parse(doc.fields.related_ids) : [];
            const siblings = doc.fields.sibling_ids ? JSON.parse(doc.fields.sibling_ids) : [];
            if (Array.isArray(related)) {
              for (const r of related) expandedDbIds.add(r);
            }
            if (Array.isArray(siblings)) {
              for (const s of siblings) expandedDbIds.add(s);
            }
          }
        } catch {
          // Fallback: continue if sync fetch is unsupported or node not found
        }
      }
    } finally {
      collection.closeSync();
    }

    // Prune using parent-absorbs-child rule
    const prunedDbIds = pruneSubtrees(Array.from(expandedDbIds));

    // --- Step 3: Collect full text for all pruned candidates ---
    //     Text = heading + summary + content (mirrors notebook: content_title + sub_title + value)
    const contentRes = await this.db.query(
      `SELECT section_id, heading, summary, content FROM section_table
       WHERE doc_id = $1 AND section_id = ANY($2)`,
      [docId, prunedDbIds]
    );

    let totalChars = 0;
    for (const row of contentRes.rows) {
      totalChars += (row.content?.length || 0) + (row.summary?.length || 0);
    }

    logger.info('Context window budget status', {
      docId,
      prunedDbIdsCount: prunedDbIds.length,
      totalChars,
      estimatedTokens: totalChars * 0.25
    });

    // --- Step 4: TF-IDF cosine re-rank — ALWAYS runs unconditionally ---
    //     Mirrors notebook: TfidfVectorizer(stop_words, ngram_range=(1,2))
    //     fitted on [query] + section_docs, then cosine_similarity(query, sections)

    const candidateTexts = contentRes.rows.map(row => ({
      section_id: row.section_id,
      // Full combined text: heading + summary + content (equivalent to notebook's
      // content_title + sub_title + value combination)
      text: [
        row.heading   || '',
        row.summary   || '',
        row.content   || ''
      ].join(' ').trim()
    }));

    // Optionally derive stopwords via LLM language detection
    const useAutoStopwords = options?.autoStopwords ?? false;
    const baseStopwords    = options?.stopwords ?? DEFAULT_STOPWORDS;
    let effectiveStopwords = baseStopwords;

    if (useAutoStopwords) {
      // LLM detects the corpus language and returns language-specific stopwords
      const { stopwords: llmStops, language, languageCode } = await this.generateLLMStopwords(
        candidateTexts.map(c => c.text)
      );
      effectiveStopwords = llmStops;
      logger.info('Using LLM-generated language-specific stopwords for TF-IDF', {
        docId, language, languageCode,
        totalStopwords: effectiveStopwords.size
      });
    }

    const { result: ranked, elapsedMs: tfidfElapsed } = await timeOperation(
      async () => tfidfCosineScore(query, candidateTexts, effectiveStopwords)
    );

    const finalDbIds = ranked.slice(0, tfidfLimit).map(r => r.section_id);

    logger.info('TF-IDF cosine re-ranking completed', {
      docId,
      tfidfElapsedMs: tfidfElapsed,
      candidateCount: candidateTexts.length,
      kept: finalDbIds.length,
      autoStopwords: useAutoStopwords,
      topScores: ranked.slice(0, tfidfLimit).map(r => ({ id: r.section_id, score: r.score.toFixed(4) }))
    });

    // --- 3.4 Recursive heading lineage climbing ---
    const { result: treeRes, elapsedMs: climbElapsed } = await timeOperation(
      async () => {
        return await this.db.query(
          `WITH RECURSIVE ancestors AS (
             SELECT id, doc_id, section_id, parent_id, level, heading, content
             FROM section_table WHERE doc_id = $1 AND section_id = ANY($2)
             UNION ALL
             SELECT s.id, s.doc_id, s.section_id, s.parent_id, s.level, s.heading, s.content
             FROM section_table s JOIN ancestors a ON s.doc_id = a.doc_id AND s.section_id = a.parent_id
           )
           SELECT DISTINCT id, section_id, parent_id, level, heading, content
           FROM ancestors ORDER BY level ASC, section_id;`,
          [docId, finalDbIds]
        );
      }
    );

    let contextStr = '';
    for (const row of treeRes.rows) {
      const cleanSecId = this.fromDbId(row.section_id);
      const level = row.level || 1;
      const heading = row.heading || '';

      const prefix = `> ${'#'.repeat(level)}${cleanSecId} ${heading}`;
      contextStr += prefix + '\n';

      if (row.content) {
        contextStr += row.content + '\n';
      }
      contextStr += '\n';
    }

    // Generate final answer
    const finalPrompt = `You are a helpful assistant answering queries based on the structured document context provided.

Here is the document context:
${contextStr}

Query: ${query}`;

    const { result: answerResponse, elapsedMs: answerElapsed } = await timeOperation(
      async () => {
        return await this.model.generate([
          { role: 'user', content: finalPrompt }
        ]);
      }
    );

    logger.info('Answer generated successfully (IHR)', {
      docId,
      path: 'tfidf',
      finalDbIdsCount: finalDbIds.length,
      contextLength: contextStr.length,
      climbElapsedMs: climbElapsed,
      answerElapsedMs: answerElapsed
    });

    return {
      answer: answerResponse.content,
      context: contextStr,
      retrievedSections: finalDbIds.map(dbId => this.fromDbId(dbId)),
      path: 'tfidf'
    };
  }
}
