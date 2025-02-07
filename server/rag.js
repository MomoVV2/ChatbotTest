// rag.js
import path from "path";
import pdfjs from "pdfjs-dist";
const { getDocument } = pdfjs;
import { promises as fs } from "fs";
import canvas from "canvas";
import { fileURLToPath } from "url";
import fetch from "node-fetch";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const { createCanvas } = canvas;
pdfjs.GlobalWorkerOptions.workerSrc = path.join(
  __dirname,
  "../node_modules/pdfjs-dist/build/pdf.worker.js"
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
    this.bankingKeywords = [
      'account', 'transfer', 'balance', 'card', 'payment', 'login',
      'security', 'transaction', 'banking', 'online', 'mobile', 'password',
      'global trust', 'bank', 'help', 'support'
    ];
  }

  addVectors(vectors, documents) {
    this.vectors.push(...vectors);
    this.documents.push(...documents);
  }

  similaritySearch(queryVector, k = 3, minScore = 0.6) {
    const scores = this.documents.map((doc, index) => ({
      score: this.cosineSimilarity(queryVector, this.vectors[index]),
      doc,
      relevance: this.getBankingRelevance(doc.pageContent)
    }));

    console.log('Search scores:', scores.map(s => ({
      score: s.score,
      content: s.doc.pageContent.substring(0, 50)
    })));

    return scores
      .filter(({ score }) => score >= minScore)
      .sort((a, b) => (b.score + b.relevance) - (a.score + a.relevance))
      .slice(0, k)
      .map(({ doc }) => doc);
  }

  getBankingRelevance(text) {
    return this.bankingKeywords.reduce((score, keyword) => 
      score + (text.toLowerCase().includes(keyword) ? 0.1 : 0), 0);
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }
}

class IntentRecognizer {
  constructor() {
    this.intents = new Map();
    this.faqData = new Map();
    this.setupBankingIntents();
  }

  async setupBankingIntents() {
    await this.addIntent('password_change', [
      'how do I change my password',
      'change password',
      'reset password',
      'update password'
    ], 'Settings→Security→Change Password→Enter new password→Confirm to update your login credentials');

    await this.addIntent('account_info', [
      'how do I check my account balance',
      'view my account',
      'account information',
      'check balance'
    ], 'Login→MyAccounts→Balance to check your account information');

    await this.addIntent('transfer_money', [
      'how do I transfer money',
      'send money',
      'make a transfer',
      'transfer funds'
    ], 'Login→Transfers→NewTransfer to send money between accounts');

    await this.addIntent('card_services', [
      'credit card issues',
      'debit card help',
      'card services',
      'block my card'
    ], 'Login→Cards→Manage to handle card-related services');
  }

  async addIntent(intentName, examples, answer) {
    const embeddings = await Promise.all(
      examples.map((example) => getEmbedding(example))
    );

    const averageEmbedding = embeddings[0].map((_, i) =>
      embeddings.reduce((sum, e) => sum + e[i], 0) / embeddings.length
    );

    this.intents.set(intentName, {
      embedding: averageEmbedding,
      answer,
    });
  }

  async addFAQIntent(question, answer) {
    const intentName = `faq_${question.toLowerCase().replace(/\W+/g, '_')}`;
    await this.addIntent(intentName, [question], answer);
    this.faqData.set(intentName, { question, answer });
  }

  async detectIntent(query, threshold = 0.75) {
    const queryEmbedding = await getEmbedding(query);
    let bestMatch = { name: null, score: -1 };

    for (const [name, intent] of this.intents.entries()) {
      const score = this.cosineSimilarity(queryEmbedding, intent.embedding);
      console.log(`Intent match: ${name} = ${score}`);
      if (score > bestMatch.score) {
        bestMatch = { name, score };
      }
    }

    return bestMatch.score >= threshold ? bestMatch : null;
  }

  async generateDynamicAnswer(intent, contextDocs, query) {
    if (intent.name.startsWith('faq_')) {
      const faqData = this.faqData.get(intent.name);
      if (faqData) {
        return [faqData.answer];
      }
    }

    const context = contextDocs.map(d => d.pageContent).join(' ');
    
    // New improved prompt for better responses
    const prompt = `[INST] You are a friendly banking assistant for Hanseatic Bank.
If you don't have specific information about a product or service:
1. Acknowledge the question politely
2. Direct them to hanseaticbank.de for detailed information
3. Offer to help with other banking matters
4. Keep response friendly and professional

For example:
"I don't have detailed information about that specific product, but you can find all details at hanseaticbank.de→Products. 
Meanwhile, I can help you with online banking, transfers, or other services!"

Question: ${query}
Context: ${context}
[/INST]`;

    const response = await fetch('https://deep.momoh.de/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-r1:1.5b',
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7, // Increased for more natural responses
          num_predict: 150, // Increased for fuller responses
          repeat_penalty: 1.2,
          top_k: 40,
          top_p: 0.9,
          mirostat: 1,
          mirostat_tau: 4.0,
          mirostat_eta: 0.1
        }
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
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
      .replace(/[^→\w\s]/gi, "")
      .replace(/\s+/g, " ")
      .replace(/\s→/g, "→")
      .substring(0, 80)
      .trim();
  }

  splitIntoMessages(text) {
    return [text];
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val ** 2, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val ** 2, 0));
    return normA && normB ? dot / (normA * normB) : 0;
  }
}

async function getEmbedding(text) {
  const response = await fetch("https://deep.momoh.de/api/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "nomic-embed-text",
      prompt: text,
    }),
  });

  if (!response.ok) {
    throw new Error(`Embedding error: ${response.status} - ${await response.text()}`);
  }

  const data = await response.json();
  return data.embedding;
}

async function loadFile(filePath) {
  try {
    console.log(`Loading file: ${filePath}`);
    const text = await fs.readFile(filePath, 'utf-8');

    if (path.basename(filePath).toLowerCase() === 'faq.txt') {
      console.log('Processing FAQ file');
      const qaPairs = text.split('\n\n')
        .filter(qa => qa.trim() && qa.includes('Q:') && qa.includes('A:'));

      console.log(`Found ${qaPairs.length} QA pairs`);
      
      return qaPairs.map(pair => {
        const [questionPart, answerPart] = pair.split('\nA:');
        const question = questionPart.replace('Q:', '').trim();
        const answer = answerPart.trim();

        return new Document({
          pageContent: `Question: ${question}\nAnswer: ${answer}`,
          metadata: {
            source: 'FAQ',
            type: 'qa',
            question: question,
            answer: answer
          }
        });
      });
    }

    if (path.extname(filePath) === '.pdf') {
      const data = new Uint8Array(await fs.readFile(filePath));
      const pdf = await getDocument({
        data,
        useSystemFonts: true,
        disableFontFace: true,
      }).promise;

      let text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map((item) => item.str).join(' ');
      }
      return splitText(text).map(content => new Document({
        pageContent: content,
        metadata: { source: path.basename(filePath), type: 'content' }
      }));
    }

    return splitText(text).map(content => new Document({
      pageContent: content,
      metadata: { source: path.basename(filePath), type: 'content' }
    }));
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
    return [];
  }
}

function splitText(text) {
  const chunks = [];
  const qaMatches = text.matchAll(/(Q:\s*.+?\s*A:\s*.+?)(?=\nQ:|$)/gis);
  for (const match of qaMatches) {
    chunks.push(match[0].trim().replace(/\n+/g, ' '));
  }

  if (chunks.length === 0) {
    const sections = text.split(/(?=\n#+ )|\n\s*\n/);
    sections.forEach((section) => {
      const clean = section.trim().replace(/\n+/g, ' ');
      if (clean.length > 50) chunks.push(clean);
    });
  }

  return chunks;
}

async function setupVectorStore() {
  const vectorStore = new SimpleVectorStore();
  const intentRecognizer = new IntentRecognizer();
  const knowledgePath = path.join(__dirname, "..", "knowledge");

  try {
    console.log('Loading knowledge from:', knowledgePath);
    const files = await fs.readdir(knowledgePath);
    console.log('Found files:', files);

    const validFiles = files.filter((f) =>
      [".txt", ".md", ".pdf"].includes(path.extname(f).toLowerCase())
    );

    for (const file of validFiles) {
      console.log(`Processing ${file}`);
      const docs = await loadFile(path.join(knowledgePath, file));
      console.log(`Loaded ${docs.length} documents from ${file}`);

      for (const doc of docs) {
        try {
          const embedding = await getEmbedding(doc.pageContent);
          vectorStore.addVectors([embedding], [doc]);

          if (doc.metadata.type === 'qa') {
            await intentRecognizer.addFAQIntent(
              doc.metadata.question,
              doc.metadata.answer
            );
          }
        } catch (e) {
          console.error(`Error embedding doc from ${file}:`, e.message);
        }
      }
    }

    console.log(`Vector store initialized with ${vectorStore.documents.length} documents`);
    console.log(`Intent recognizer initialized with ${intentRecognizer.intents.size} intents`);
    
    return { vectorStore, intentRecognizer };
  } catch (error) {
    console.error("Vector store initialization failed:", error);
    throw error;
  }
}

export {
  setupVectorStore,
  SimpleVectorStore,
  IntentRecognizer,
  getEmbedding,
};
