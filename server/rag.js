import path from 'path';
import pdfjs from 'pdfjs-dist';
const { getDocument } = pdfjs;
import { promises as fs } from 'fs';
import canvas from 'canvas';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';

// ES Module __dirname polyfill
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure PDF.js for Node.js
const { createCanvas } = canvas;
// Replace the worker config with:
pdfjs.GlobalWorkerOptions.workerSrc = path.join(
  __dirname,
  '../node_modules/pdfjs-dist/build/pdf.worker.js'
);

class Document {
  constructor({ pageContent, metadata }) {
    this.pageContent = pageContent;
    this.metadata = metadata || {};
  }
}

class SimpleVectorStore {
  constructor() {
    this.vectors = [];
    this.documents = [];
  }

  addVectors(vectors, documents) {
    this.vectors.push(...vectors);
    this.documents.push(...documents);
  }

  similaritySearch(queryVector, k = 3, minScore = 0.75) {
    const scores = this.documents.map((doc, index) => ({
      score: this.cosineSimilarity(queryVector, this.vectors[index]),
      doc
    }));

    return scores
      .filter(({ score }) => score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + (val * (b[i] || 0)), 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }
}

class IntentRecognizer {
  constructor() {
    this.intents = new Map();
  }

  async addIntent(intentName, examples, answer) {
    const embeddings = await Promise.all(
      examples.map(example => getEmbedding(example))
    );
    
    const averageEmbedding = embeddings[0].map((_, i) =>
      embeddings.reduce((sum, e) => sum + e[i], 0) / embeddings.length
    );
    
    this.intents.set(intentName, {
      embedding: averageEmbedding,
      answer
    });
  }

  async detectIntent(query, threshold = 0.85) {
    const queryEmbedding = await getEmbedding(query);
    let bestMatch = { name: null, score: -1 };

    for (const [name, intent] of this.intents.entries()) {
      const score = this.cosineSimilarity(queryEmbedding, intent.embedding);
      if (score > bestMatch.score) {
        bestMatch = { name, score };
      }
    }

    return bestMatch.score >= threshold ? bestMatch : null;
  }

  async generateDynamicAnswer(intent, contextDocs, query) {
    const context = contextDocs.map(d => d.pageContent).join('\n');
    
    const prompt = `<|im_start|>system
    You are a mobile-first support bot. Respond ONLY in this format:
    •[Action] [Details] (max 7 words)
    Max 3 bullet points. No explanations.<|im_end|>
    <|im_start|>user
    ${query}<|im_end|>
    <|im_start|>assistant
    `;
  
    const response = await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-r1:1.5b',
        prompt: prompt,
        options: {
          temperature: 0.3,
          max_tokens: 80,
          repeat_penalty: 2.5
        }
      })
    });

    if (!response.ok) {
      throw new Error(`LLM API error: ${response.status} - ${await response.text()}`);
    }

    const data = await response.json();
    return this.processResponse(data.response);
  }

  processResponse(text) {
    const cleaned = this.cleanResponse(text);
    return this.splitIntoMessages(cleaned);
  }

  cleanResponse(text) {
    return text
      .replace(/[^•\w\s]/g, '') // Remove all punctuation
      .replace(/\s+/g, ' ')     // Single spaces only
      .substring(0, 80);        // Hard character limit
  }
  

  splitIntoMessages(text) {
    return text.split('•')
      .filter(b => b.trim())
      .map(b => `• ${b.trim()}`);
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }
}

async function getEmbedding(text) {
  const response = await fetch('http://127.0.0.1:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'nomic-embed-text',
      prompt: text
    })
  });
  
  if (!response.ok) {
    throw new Error(`Embedding error: ${response.status} - ${await response.text()}`);
  }

  const data = await response.json();
  return data.embedding;
}

function splitText(text) {
  const chunks = [];
  // First try QA pattern
  const qaMatches = text.matchAll(/(Q:\s*.+?\s*A:\s*.+?)(?=\nQ:|$)/gis);
  for (const match of qaMatches) {
    chunks.push(match[0].trim().replace(/\n+/g, ' '));
  }

  // Fallback to section splitting
  if (chunks.length === 0) {
    const sections = text.split(/(?=\n#+ )|\n\s*\n/);
    sections.forEach(section => {
      const clean = section.trim().replace(/\n+/g, ' ');
      if (clean.length > 50) chunks.push(clean);
    });
  }

  return chunks;
}

async function loadFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  try {
    let text;
    
    if (ext === '.pdf') {
      const data = new Uint8Array(await fs.readFile(filePath));
      const pdf = await getDocument({
        data,
        useSystemFonts: true,
        disableFontFace: true,
      }).promise;

      text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(item => item.str).join(' ');
      }
    } else if (ext === '.txt' || ext === '.md') {
      text = await fs.readFile(filePath, 'utf-8');
    } else {
      return [];
    }

    return splitText(text).map(content => 
      new Document({
        pageContent: content,
        metadata: { 
          source: path.basename(filePath),
          isQA: content.startsWith('Q:')
        }
      })
    );
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return [];
  }
}

async function setupVectorStore() {
  const vectorStore = new SimpleVectorStore();
  const intentRecognizer = new IntentRecognizer();
  const knowledgePath = path.join(__dirname, '..', 'knowledge');

  try {
    const files = await fs.readdir(knowledgePath);
    const validFiles = files.filter(f => 
      ['.txt', '.md', '.pdf'].includes(path.extname(f).toLowerCase())
    );

    for (const file of validFiles) {
      const docs = await loadFile(path.join(knowledgePath, file));
      for (const doc of docs) {
        try {
          const embedding = await getEmbedding(doc.pageContent);
          vectorStore.addVectors([embedding], [doc]);
        } catch (e) {
          console.error(`Error embedding doc from ${file}:`, e.message);
        }
      }
    }
    
    console.log(`Vector store initialized with ${vectorStore.documents.length} documents`);
    return { vectorStore, intentRecognizer };
  } catch (error) {
    console.error('Vector store initialization failed:', error);
    throw error;
  }
}

export { 
  setupVectorStore,
  SimpleVectorStore,
  IntentRecognizer,
  getEmbedding
};