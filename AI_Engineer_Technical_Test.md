# 🧪 AI Engineer Technical Test — Chatbot with RAG + Fine-Tuning + API

## 🎯 Objective

Build a **chatbot AI assistant** that:

1. Uses **RAG (Retrieval-Augmented Generation)** on custom data
2. Demonstrates **model adaptation (fine-tuning OR structured prompting)**
3. Exposes a **REST API** for interaction

---

## 📦 Problem Statement

You are required to build an AI assistant for a company (e.g., SaaS / ERP / support system).

The chatbot should:

* Answer questions based on **custom documents**
* Provide **accurate, context-aware responses**
* Be accessible via API

---

## 🧱 Requirements

---

## 1. Data Preparation

Use any of the following as your knowledge base:

* PDF documents
* Text files
* Website content
* Internal docs (mock data allowed)

You must:

* Clean and structure the data
* Split into chunks
* Store embeddings in a vector store

---

## 2. RAG Implementation

Build a RAG pipeline:

* Convert text → embeddings
* Store in vector DB (e.g., FAISS / Chroma)
* Retrieve relevant chunks based on query
* Pass context to LLM

Expected flow:

```
User asks → Retrieve relevant docs → Inject into prompt → Generate answer
```

---

## 3. Model Adaptation (IMPORTANT)

You must implement **one of the following**:

### Option A — Fine-Tuning

* Fine-tune a small model (or simulate with small dataset)
* Example: Q&A pairs from your domain

### Option B — Prompt Engineering (acceptable for junior)

* Create structured system prompts
* Add rules like:
  * Answer only from context
  * Say “I don’t know” if not found

---

## 4. API Development

Expose the chatbot via API using **FastAPI**

### Required Endpoints:

#### POST /chat

Input:

```json
{
  "message": "What is your refund policy?"
}
```

Output:

```json
{
  "response": "...",
  "sources": ["doc1", "doc2"]
}
```

---

#### POST /ingest

* Upload new documents dynamically

---

## 5. Project Structure

Expected structure:

```
/app
  /rag
  /api
  /models
  /data
main.py
requirements.txt
README.md
```

---

## ⭐ Bonus

* Streaming responses
* Chat history memory
* Use of **LangChain** or similar
* Dockerized app (**Docker**)
* Evaluation (accuracy of answers)
* Logging + error handling

---

## ⏱ Time Limit

* 1–2 Days

---

## 📦 Deliverables

* Source code (GitHub repo)
* README with:
  * Setup instructions
  * How to run API
  * Example requests
* Sample dataset used

---

## 🎯 Expected Outcome

A working chatbot API that:

* Answers questions from custom data
* Uses RAG correctly
* Is deployable and testable

---
