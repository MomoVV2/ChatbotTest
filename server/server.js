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
        greeting: "Hi there! I'm your Hanseatic Helper, ready to assist with all your banking needs!",
        identity: "I'm your Hanseatic Helper - I'm here to help you with banking, transfers, accounts, and anything else you need!",
        services: "I handle everything from account management and transfers to cards and security. What can I help you with?",
        capabilities: "I'm your go-to for all banking needs - transfers, accounts, cards, you name it! What's on your mind?",
        default: "Let me help you with that! What would you like to know?"
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
        greeting: "Hi there! I'm your Hanseatic Helper, ready to assist with all your banking needs!",
        identity: "I'm your Hanseatic Helper - I'm here to help you with banking, transfers, accounts, and anything else you need!",
        services: "I handle everything from account management and transfers to cards and security. What can I help you with?",
        capabilities: "I'm your go-to for all banking needs - transfers, accounts, cards, you name it! What's on your mind?",
        default: "Let me help you with that! What would you like to know?"
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


const commonQuestions = {
  'who are you': "Hi! I'm your Hanseatic Helper, a banking assistant designed to help you with financial services and banking needs. I can help you with accounts, transfers, and much more!",
  'what are you': "I'm an AI banking assistant created to help customers with their banking needs. I can assist you with everything from account management to transfers and security features!",
  'what do you do': "I help customers with their banking needs! Whether it's managing accounts, making transfers, or handling card services - I'm here to assist you. What can I help you with?",
  'features': "We offer several key features including online banking, secure transfers, card management, and account services. Which area would you like to know more about?",
  'password': "To change your password, go to Settings→Security→Change Password. You'll need to enter your current password, then your new one. Need help with the process?",
};

app.post('/ask', async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Transfer-Encoding', 'chunked');

  const { question } = req.body;
  if (!question) {
    res.write(JSON.stringify({ message: "Could you please ask your question again?" }));
    return res.end();
  }

  try {
    const questionLower = question.toLowerCase().trim();
    let response;

    // Handle basic identity and location questions directly
    if (questionLower.includes('who are you')) {
      response = "Hi! I'm your Hanseatic Helper, a banking assistant designed to help you with financial services and banking needs. I can help you with accounts, transfers, and much more!";
    } 
    else if (questionLower.match(/where|location|address|where.*you.*at/)) {
      response = "I'm a digital banking assistant for Hanseatic Bank. While I'm available 24/7 through this app, you can find our physical locations and contact details at hanseaticbank.de→Contact.";
    }
    else if (questionLower.includes('features') || questionLower.includes('what can you do')) {
      response = "I can help you with online banking, account management, transfers, card services, and security features. Which of these would you like to know more about?";
    }
    else {
      // Check FAQ for relevant information
      const queryEmbedding = await getEmbedding(question);
      const results = await vectorStore.similaritySearch(queryEmbedding, 1);
      
      if (results.length > 0 && results[0].metadata.type === 'qa') {
        response = results[0].metadata.answer;
      } else {
        // For unknown questions, use AI with a strict timeout
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);

        try {
          const aiResponse = await fetch('https://deep.momoh.de/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: 'deepseek-r1:1.5b',
              prompt: `[INST] You are the Hanseatic Helper, a banking assistant.
Question: "${question}"
- Answer directly and helpfully
- You are part of Hanseatic Bank
- Keep responses under 2 sentences
- Suggest specific banking services when relevant
[/INST]`,
              stream: false,
              options: {
                temperature: 0.3,
                num_predict: 80,
                repeat_penalty: 1.5,
                top_k: 40,
                top_p: 0.9
              }
            }),
            signal: controller.signal
          });

          const data = await aiResponse.json();
          response = data.response
            .replace(/<think>[\s\S]*?<\/think>/g, '')
            .replace(/\n+/g, ' ')
            .trim();
        } catch (error) {
          // If AI times out, give a specific helpful response based on question context
          if (questionLower.includes('card')) {
            response = "I can help you with card services including applications, limits, and security features. Which aspect would you like to know more about?";
          } else if (questionLower.includes('account')) {
            response = "I can assist you with account management, balances, and settings. What specific information do you need?";
          } else {
            response = "I can help you with online banking, transfers, cards, and account services. Which area interests you?";
          }
        } finally {
          clearTimeout(timeout);
        }
      }
    }

    res.write(JSON.stringify({ response }));
    res.end();

  } catch (error) {
    console.error('ERROR:', error);
    res.write(JSON.stringify({ 
      response: "I can help you with banking services like transfers, accounts, and cards. What would you like to know?" 
    }));
    res.end();
  }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
