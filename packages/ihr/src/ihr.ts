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

export interface IngestSection {
  section_id: string;
  parent_id: string | null;
  level: number;
  heading: string;
  content: string; // full text for leaf nodes
  summary: string; // 2-3 sentences summary + keywords
}

export interface RetrieveOptions {
  limit?: number;
  tfidfLimit?: number;
  alwaysRunTfidf?: boolean;
}

export interface RetrieveResponse {
  answer: string;
  context: string;
  retrievedSections: string[];
  path: 'fast' | 'tfidf';
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

/**
 * Tokenizes text by lowercasing, stripping punctuation, and removing English stopwords.
 */
export function tokenize(text: string): string[] {
  const stopwords = new Set([
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
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 0 && !stopwords.has(t));
}

/**
 * Calculates TF-IDF scores for full text items relative to query terms.
 */
export function tfidfScore(
  query: string,
  sections: { section_id: string; text: string }[]
): { section_id: string; score: number }[] {
  const qterms = tokenize(query);
  if (qterms.length === 0 || sections.length === 0) {
    return sections.map(s => ({ section_id: s.section_id, score: 0 }));
  }

  // 1. Pre-tokenize all section texts
  const tokenizedSections = sections.map(s => ({
    section_id: s.section_id,
    terms: tokenize(s.text)
  }));

  // 2. Compute IDF for each query term
  const idf: Record<string, number> = {};
  for (const t of qterms) {
    const matchingDocsCount = tokenizedSections.filter(s => s.terms.includes(t)).length;
    idf[t] = Math.log(sections.length / (1 + matchingDocsCount));
  }

  // 3. Score each section
  const scored = tokenizedSections.map(({ section_id, terms }) => {
    if (terms.length === 0) return { section_id, score: 0 };
    let score = 0;
    for (const t of qterms) {
      const termFreq = terms.filter(x => x === t).length;
      const tf = termFreq / terms.length;
      score += tf * (idf[t] || 0);
    }
    return { section_id, score };
  });

  return scored.sort((a, b) => b.score - a.score);
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

    const mapResponse = await this.model.generate([
      { role: 'user', content: outlinePrompt }
    ], { config: new ChatConfig({ jsonMode: true }) });

    let relationshipMap: Record<string, { related_ids?: string[], sibling_ids?: string[], prerequisite_ids?: string[] }> = {};
    try {
      relationshipMap = JSON.parse(mapResponse.content);
    } catch (err) {
      console.warn('Could not parse outline structure LLM response. Defaulting relationships to empty.', err);
    }

    // 5. Generate embeddings of enriched section text (Title + Heading + Body snippet)
    const richTexts = sections.map(s => {
      const bodySnippet = s.summary || s.content.slice(0, 200);
      return `${title} — ${s.heading}. ${bodySnippet}`;
    });
    const embeddingsArray = await this.embeddings.embedDocuments(richTexts);

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
    } finally {
      collection.closeSync();
    }
  }

  /**
   * Helper to perform similarity search over the zvec collection.
   */
  private async performSimilaritySearch(docId: string, query: string, limit: number): Promise<string[]> {
    const queryEmbedding = await this.embeddings.embedQuery(query);
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
    const limit = options?.limit ?? 5;
    const tfidfLimit = options?.tfidfLimit ?? 3;
    const alwaysRunTfidf = options?.alwaysRunTfidf ?? false;

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
    let response = await this.model.generate(messages, {
      tools: [searchTool],
      config: new ChatConfig({ jsonMode: true })
    });

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

      response = await this.model.generate(messages, {
        tools: [searchTool],
        config: new ChatConfig({ jsonMode: true })
      });
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

    // --- 3.2 Context Window Gate ---
    const lenRes = await this.db.query(
      `SELECT section_id, length(content) AS chars FROM section_table WHERE doc_id = $1 AND section_id = ANY($2)`,
      [docId, prunedDbIds]
    );

    let totalChars = 0;
    for (const row of lenRes.rows) {
      totalChars += parseInt(row.chars) || 0;
    }

    const estimatedTokens = totalChars * 0.25;
    let finalDbIds = prunedDbIds;
    let pathUsed: 'fast' | 'tfidf' = 'fast';

    if (alwaysRunTfidf || estimatedTokens > 6000) {
      pathUsed = 'tfidf';
      // Fetch full content texts for candidates
      const contentRes = await this.db.query(
        `SELECT section_id, heading, content FROM section_table WHERE doc_id = $1 AND section_id = ANY($2)`,
        [docId, prunedDbIds]
      );
      
      const candidateTexts = contentRes.rows.map(row => ({
        section_id: row.section_id,
        text: `${row.heading || ''} ${row.content || ''}`
      }));

      const ranked = tfidfScore(query, candidateTexts);
      finalDbIds = ranked.slice(0, tfidfLimit).map(r => r.section_id);
    }

    // --- 3.4 Recursive heading lineage climbing ---
    const treeRes = await this.db.query(
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

    const answerResponse = await this.model.generate([
      { role: 'user', content: finalPrompt }
    ]);

    return {
      answer: answerResponse.content,
      context: contextStr,
      retrievedSections: finalDbIds.map(dbId => this.fromDbId(dbId)),
      path: pathUsed
    };
  }
}
