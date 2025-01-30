const { Ollama } = require("@langchain/community/llms/ollama");

async function test() {
  const llm = new Ollama({
    baseUrl: "http://127.0.0.1:11434",
    model: "mistral"
  });

  const response = await llm.invoke("What is 2+2?");
  console.log("Test response:", response);
}

test();