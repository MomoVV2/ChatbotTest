# ProjectKI

This repository implements a lightweight, custom vector-based information retrieval system using a RESTful API with Node.js and Express. It leverages embeddings and cosine similarity to retrieve and rank relevant documents. The Ollama server is used for generating embeddings and answers.

---

## Features
- Custom vector store for document embeddings.
- Retrieval of top-N relevant documents based on cosine similarity.
- Integration with Ollama server for embeddings and language model responses.
- RESTful API endpoints for interacting with the system.

---

## Prerequisites

Ensure you have the following installed:

1. **Node.js** (v18 or higher)
2. **npm** (bundled with Node.js)
3. **Ollama Server**
   - Install Ollama ([Ollama Documentation](https://www.ollama.ai))
   - Ensure the server is running on `http://127.0.0.1:11434`.
4. Install the required model(s) in Ollama:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/projectki.git
   cd projectki
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Ollama server if it is not already running:
   ```bash
   ollama start
   ```

---

## Usage

### Start the Server
Run the server:
```bash
node server.js
```
Server will be running on `http://localhost:3000`.

### API Endpoints

#### 1. `/ask`
**Method:** POST

**Description:** Retrieve an answer based on a user query.

**Request Body:**
```json
{
  "question": "Your question here"
}
```

**Response:**
```json
{
  "answer": "Generated response here"
}
```

### Example with `curl`:
```bash
curl -X POST http://localhost:3000/ask \
-H "Content-Type: application/json" \
-d '{"question": "What is the capital of France?"}'
```

---

## Code Overview

### `server.js`
- Sets up an Express.js server.
- Initializes a custom in-memory vector store with document embeddings.
- Handles embedding generation via Ollama server.
- Implements a cosine similarity function for document retrieval.
- Provides the `/ask` endpoint for user queries.

### Key Functions
1. **`getEmbedding(text)`**
   - Generates embeddings for input text using the Ollama API.

2. **`cosineSimilarity(a, b)`**
   - Computes the cosine similarity