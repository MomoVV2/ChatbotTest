const path = require('path');
const fs = require('fs').promises;
const pdfParse = require('pdf-parse');

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
    
    const prompt = `[STRICT FORMATTING RULES]
- Answer directly without greetings
- Use ONLY "•" for bullet points
- Max 3 bullets per section
- Max 200 characters per message part
- NEVER use markdown or links
- Separate sections with blank lines

Context: ${context}

Question: ${query}

Concise, split answer:`;

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama2',
        prompt: prompt,
        stream: false,
        options: {
          max_tokens: 1200,
          temperature: 0.3
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
      .replace(/[*-]>\s?/g, '• ')   // Normalize bullets
      .replace(/\[\d+\]/g, '')      // Remove citations
      .replace(/\(http\S+\)/g, '')  // Remove URLs
      .replace(/\n{3,}/g, '\n\n')   // Limit newlines
      .replace(/^\s*[\r\n]/gm, '')  // Remove empty lines
      .trim();
  }

  splitIntoMessages(text) {
    const MAX_LENGTH = 200;
    const messages = [];
    let currentMessage = '';
    
    const sections = text.split(/(?:\n\s*){2,}/);
    
    for (const section of sections) {
      const lines = section.split('\n');
      
      for (const line of lines) {
        if ((currentMessage + line).length > MAX_LENGTH) {
          messages.push(currentMessage.trim());
          currentMessage = '';
        }
        currentMessage += line + '\n';
        
        // Split at natural breaks
        if (line.startsWith('•') && currentMessage.length > MAX_LENGTH/2) {
          messages.push(currentMessage.trim());
          currentMessage = '';
        }
      }
      
      if (currentMessage.length > 0) {
        messages.push(currentMessage.trim());
        currentMessage = '';
      }
    }
    
    return messages.filter(m => m.length > 0);
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }
}

async function getEmbedding(text) {
  const response = await fetch('http://localhost:11434/api/embeddings', {
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
      const dataBuffer = await fs.readFile(filePath);
      const pdfData = await pdfParse(dataBuffer);
      text = pdfData.text;
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

module.exports = { 
  setupVectorStore,
  SimpleVectorStore,
  IntentRecognizer,
  getEmbedding
};
