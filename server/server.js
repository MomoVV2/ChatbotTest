import express from 'express';
import cors from 'cors';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { setupVectorStore, getEmbedding } from './rag.js';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

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
      name: "Hanseatic Helper",
      style: "Friendly and professional banking assistant",
      introduction: "",
      services: [],
      responseGuidelines: [],
      responses: {
        greeting: "Welcome! I'm the Hanseatic Helper. How can I assist you with your banking needs today?",
        identity: "I'm the Hanseatic Helper, your dedicated banking assistant focused on providing professional financial support.",
        services: "I assist with account management, transactions, online banking, card services, and other banking-related matters.",
        capabilities: "I can help you with account services, transfers, security features, and general banking questions.",
        default: "How can I assist you with your banking needs today?"
      }
    };
    
    let currentSection = '';
    lines.forEach(line => {
      if (line.startsWith('name:')) {
        persona.name = line.replace('name:', '').trim();
      } else if (line.startsWith('style:')) {
        persona.style = line.replace('style:', '').trim();
      } else if (line.startsWith('introduction:')) {
        persona.introduction = line.replace('introduction:', '').trim();
      } else if (line.startsWith('services:')) {
        currentSection = 'services';
      } else if (line.startsWith('responseGuidelines:')) {
        currentSection = 'responseGuidelines';
      } else if (line.startsWith('- ') && currentSection === 'services') {
        persona.services.push(line.replace('- ', '').trim());
      } else if (line.startsWith('- ') && currentSection === 'responseGuidelines') {
        persona.responseGuidelines.push(line.replace('- ', '').trim());
      }
    });
    
    return persona;
  } catch (error) {
    console.error('Error loading persona config:', error);
    return {
      name: "Hanseatic Helper",
      style: "Friendly and professional banking assistant",
      responses: {
        greeting: "Welcome! I'm the Hanseatic Helper. How can I assist you with your banking needs today?",
        identity: "I'm the Hanseatic Helper, your dedicated banking assistant.",
        services: "I assist with banking services and financial matters.",
        capabilities: "I can help you with banking-related questions and services.",
        default: "How can I assist you with your banking needs today?"
      }
    };
  }
}

async function initializeSystem() {
  try {
    AI_PERSONA = await loadPersonaConfig();
    const ragSystem = await setupVectorStore();
    vectorStore = ragSystem.vectorStore;
    intentRecognizer = ragSystem.intentRecognizer;
    
    console.log("RAG system ready");
    console.log("Loaded personality:", AI_PERSONA);
  } catch (error) {
    console.error("System initialization failed:", error);
    process.exit(1);
  }
}

initializeSystem().catch(error => {
  console.error("Critical initialization failed:", error);
  process.exit(1);
});

function getPersonaResponse(question) {
  question = question.toLowerCase();
  
  if (question.includes('who are you') || question.includes('your name')) {
    return AI_PERSONA.responses.identity;
  }
  if (question.includes('what do you do') || question.includes('your job')) {
    return AI_PERSONA.responses.services;
  }
  if (question.includes('help') || question.includes('can you')) {
    return AI_PERSONA.responses.capabilities;
  }
  if (question.length < 10 || question.includes('hi') || question.includes('hello')) {
    return AI_PERSONA.responses.greeting;
  }
  
  return null;
}

app.post('/ask', async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Transfer-Encoding', 'chunked');

  if (!vectorStore || !intentRecognizer) {
    res.write(JSON.stringify({ message: "System initializing..." }) + "\n");
    return res.end();
  }

  const { question, conversation } = req.body;
  if (!question) {
    res.write(JSON.stringify({ message: "Error: Question required" }) + "\n");
    return res.end();
  }

  try {
    // First check for basic questions that have predefined responses
    const questionLower = question.toLowerCase();
    let response;

    if (questionLower.includes('who are you') || questionLower.includes('your name')) {
      response = "I'm the Hanseatic Helper, your dedicated banking assistant focused on providing professional financial support.";
    } 
    else if (questionLower.includes('what do you do') || questionLower.includes('what can you do')) {
      response = "I assist with account management, online banking, transactions, card services, and other banking-related matters.";
    }
    else if (questionLower.includes('credit card') || questionLower.includes('card')) {
      response = "Our credit cards offer various benefits! Visit hanseaticbank.de→Cards for details, or I can help you with specific features and applications.";
    }
    
    if (response) {
      res.write(JSON.stringify({ response }));
      return res.end();
    }

    // If no predefined response, generate a contextual one
    const prompt = `[INST] You are the Hanseatic Helper, a professional banking assistant.
Respond directly to: "${question}"

Rules:
- Stay in character as a banking assistant
- Be professional but friendly
- If unsure, suggest checking hanseaticbank.de
- Keep responses clear and helpful
- No thinking out loud
[/INST]`;

    const aiResponse = await fetch('https://deep.momoh.de/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'deepseek-r1:1.5b',
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          num_predict: 100,
          repeat_penalty: 1.5,
          top_k: 40,
          top_p: 0.9,
          mirostat: 1,
          mirostat_tau: 4.0,
          mirostat_eta: 0.1,
          stop: ["<think>", "</think>", "[INST]", "[/INST]"]
        }
      })
    });

    if (!aiResponse.ok) {
      throw new Error(`API error: ${aiResponse.status}`);
    }

    const data = await aiResponse.json();
    let finalResponse = data.response || '';
    
    // Clean up the response
    finalResponse = finalResponse
      .replace(/<think>[\s\S]*?<\/think>/g, '')
      .replace(/\n+/g, ' ')
      .replace(/^(Let me|I need to|I should|Okay|Hmm|Well)/i, '')
      .replace(/^(Hello|Hi|Hey|Greetings)[,!]?\s*/i, '')
      .trim();

    if (!finalResponse || finalResponse.length < 2) {
      finalResponse = "I can help you with your banking needs. For specific product information, please visit hanseaticbank.de.";
    }

    res.write(JSON.stringify({ response: finalResponse }));
    res.end();

  } catch (error) {
    console.error('ERROR:', error);
    res.write(JSON.stringify({ message: "An unexpected error occurred. Please try again." }) + "\n");
    res.end();
  }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
