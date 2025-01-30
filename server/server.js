import express from 'express';
import cors from 'cors';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { setupVectorStore, getEmbedding } from './rag.js';
import fetch from 'node-fetch';

// ES Module __dirname polyfill
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// Configuration paths
const DATA_DIR = path.join(__dirname, '..');
const PERSONA_FILE = path.join(DATA_DIR, 'knowledge', 'persona.txt');
const FAQ_FILE = path.join(DATA_DIR, 'knowledge', 'faq.txt');

let vectorStore;
let intentRecognizer;
let AI_PERSONA = {};

async function loadPersonaConfig() {
  try {
    const data = await fs.readFile(PERSONA_FILE, 'utf-8');
    const lines = data.split('\n').filter(line => line.trim() !== '');
    
    const persona = {
      name: "Default Assistant",
      style: "Helpful technical support",
      responseGuidelines: ["Provide clear, concise answers"]
    };
    
    let currentKey = '';
    
    lines.forEach(line => {
      if (line.startsWith('name:')) {
        persona.name = line.replace('name:', '').trim();
      } else if (line.startsWith('style:')) {
        persona.style = line.replace('style:', '').trim();
      } else if (line.startsWith('responseGuidelines:')) {
        currentKey = 'responseGuidelines';
        persona.responseGuidelines = [];
      } else if (currentKey === 'responseGuidelines' && line.startsWith('- ')) {
        persona.responseGuidelines.push(line.replace('- ', '').trim());
      }
    });

    persona.responseGuidelines = persona.responseGuidelines || [];
    return persona;
  } catch (error) {
    console.error('Error loading persona config:', error);
    return {
      name: "Default Assistant",
      style: "Helpful technical support",
      responseGuidelines: ["Provide clear, concise answers"]
    };
  }
}

async function initializeSystem() {
  try {
    AI_PERSONA = await loadPersonaConfig();
    const ragSystem = await setupVectorStore(FAQ_FILE);
    vectorStore = ragSystem.vectorStore;
    intentRecognizer = ragSystem.intentRecognizer;
    
    console.log("RAG system ready");
    console.log("Loaded personality:", AI_PERSONA.name);
  } catch (error) {
    console.error("System initialization failed:", error);
    process.exit(1);
  }
}

initializeSystem().catch(error => {
  console.error("Critical initialization failed:", error);
  process.exit(1);
});

app.post('/ask', async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Transfer-Encoding', 'chunked');

  if (!vectorStore || !intentRecognizer) {
      res.write(JSON.stringify({ message: "System initializing..." }) + "\n");
      return res.end();
  }

  const { question } = req.body;
  if (!question) {
      res.write(JSON.stringify({ message: "Error: Question required" }) + "\n");
      return res.end();
  }

  try {
      let responseSent = false;
      const intent = await intentRecognizer.detectIntent(question);
      
      if (intent) {
          const intentDetails = intentRecognizer.intents.get(intent.name);
          const intentAnswer = await generateDynamicAnswer(
              `System answer: ${intentDetails.answer}`,
              question,
              AI_PERSONA
          );

          const compactMessages = chunkLines(intentAnswer.split("\n"), 4);
          compactMessages.forEach(message => {
              res.write(JSON.stringify({ message }) + "\n");
          });
          responseSent = true;
      }

      if (!responseSent) {
          const queryEmbedding = await getEmbedding(question);
          const results = vectorStore.similaritySearch(queryEmbedding, 5);

          if (results.length > 0) {
              const context = results
                  .map(({ doc }, i) => `SOURCE ${i + 1} (${doc.metadata.source}):\n${doc.pageContent}`)
                  .join('\n\n');

              const dbAnswer = await generateDynamicAnswer(context, question, AI_PERSONA);
              const compactMessages = chunkLines(dbAnswer.split("\n"), 4);
              compactMessages.forEach(message => {
                  res.write(JSON.stringify({ message }) + "\n");
              });
              responseSent = true;
          }
      }

      if (!responseSent) {
          const fallbackAnswer = await generateDynamicAnswer(
              "I couldn’t find a match, but here’s what I think:", 
              question, 
              AI_PERSONA
          );
          const compactMessages = chunkLines(fallbackAnswer.split("\n"), 4);
          compactMessages.forEach(message => {
              res.write(JSON.stringify({ message }) + "\n");
          });
      }

      res.end();
  } catch (error) {
      console.error('ERROR:', error);
      res.write(JSON.stringify({ message: "An unexpected error occurred. Please try again." }) + "\n");
      res.end();
  }
});

function chunkLines(lines, maxLines) {
  const chunks = [];
  for (let i = 0; i < lines.length; i += maxLines) {
      chunks.push(lines.slice(i, i + maxLines).join("\n"));
  }
  return chunks;
}

async function generateDynamicAnswer(context, question, persona) {
  try {
    const safePersona = {
      name: persona.name || "Assistant",
      style: persona.style || "Helpful technical support",
      responseGuidelines: persona.responseGuidelines || ["Provide clear, concise answers"]
    };

    const guidelines = (safePersona.responseGuidelines || [])
      .map((g, i) => `${i+1}. ${g}`)
      .join('\n') || '1. Provide the best possible answer';

    const responsePrompt = `[INST] You are ${safePersona.name}, ${safePersona.style}.
Guidelines:
${guidelines}

Context Data:
${context}

User Question: ${question}

Create a helpful response that:
- STARTS DIRECTLY WITH THE ANSWER (no greetings)
- Uses ONLY these formatting elements:
  • Bullet points starting with "•"
  • Numbered lists when giving steps
  • Paragraph breaks with blank lines
- Keep paragraphs under 3 lines
- NEVER use markdown, emojis, or special formatting
- If using bullets/numbering:
  - Put each item on its own line
  - Leave a blank line after the list

Example GOOD format:
To resolve the issue:
• First do X
• Then perform Y
• Finally complete Z

For additional help:
1. Open settings
2. Navigate to section A
3. Enable option B

${context ? 'Use context where relevant' : ''}
[/INST]`;

    const response = await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'mistral',
        prompt: responsePrompt,
        stream: false,
        options: {
          temperature: 0.7,
          num_predict: 300,
          repeat_penalty: 1.5,
          top_k: 50,
          top_p: 0.9,
          mirostat: 2,
          mirostat_tau: 5.0,
          mirostat_eta: 0.1
        }
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    if (!data?.response) {
      throw new Error('Invalid response format from Ollama');
    }

    let finalText = data.response
      .replace(/\[INST\].*\[\/INST\]/gs, '')
      .trim();

    finalText = finalText
      .replace(/^(hello|hi|hey|greetings)[^\n]*\n?/gim, '')
      .replace(/^(it\s+seems\s+like|it\s+looks\s+like).*\n?/gim, '')
      .replace(/[\u{1F300}-\u{1FAFF}]/gu, '')
      .replace(/([*\-➢])/g, '•')
      .replace(/(•\s.*?)(\n+)(•)/g, '$1\n$3')
      .replace(/(•.*\n)([^\n•])/g, '$1\n\n$2')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/(.{120,}?)\s/g, '$1\n')
      .trim();

    return finalText;

  } catch (error) {
    console.error('Generation error:', error);
    return `Let's try a different approach... (${error.message})`;
  }
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));