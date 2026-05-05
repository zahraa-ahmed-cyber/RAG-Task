# 🤖 Chatbot RAG — AI Support Assistant

A production-ready chatbot API built with **RAG (Retrieval-Augmented Generation)** on two real datasets:

| Layer | Dataset | Purpose |
|---|---|---|
| **RAG knowledge base** | Stripe public docs (scraped) | Grounds answers in real documentation |
| **Fine-tuning** | Bitext Customer Support (HuggingFace) | Adapts model tone & domain understanding |

---
get your api key from here : https://console.groq.com/landing/llama-api
dataset url : https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

---

## 🏗 Architecture

```
User query
    │
    ▼
POST /chat
    │
    ├─ 1. Embed query  (sentence-transformers / OpenAI)
    ├─ 2. Search FAISS / Chroma vector store
    ├─ 3. Retrieve top-K chunks
    ├─ 4. Inject context into system prompt
    └─ 5. LLM generates grounded answer
         │
         └─► response + sources
```

---

## 📁 Project Structure

```
chatbot-rag/
├── app/
│   ├── api/
│   │   ├── chat.py          # POST /chat
│   │   ├── ingest.py        # POST /ingest
│   │   └── search.py        # GET  /search  +  GET /health
│   ├── rag/
│   │   ├── document_loader.py   # PDF / TXT / HTML parser + chunker
│   │   ├── embeddings.py        # Local (sentence-transformers) or OpenAI
│   │   ├── vector_store.py      # FAISS / Chroma wrapper
│   │   └── pipeline.py          # RAG chain + streaming
│   ├── models/
│   │   ├── finetune.py          # LoRA fine-tuning (TRL + PEFT)
│   │   └── inference.py         # Load fine-tuned model
│   └── config.py            # Pydantic settings from .env
├── scripts/
│   ├── scrape_stripe_docs.py      # Layer 1: scrape Stripe docs
│   ├── prepare_bitext_dataset.py  # Layer 2: download & process Bitext
│   ├── build_index.py             # Build FAISS index from data/raw/
│   └── evaluate.py                # Accuracy evaluation
├── data/
│   ├── raw/
│   │   ├── stripe/            # Scraped Stripe .txt files
│   │   └── bitext/            # Bitext intent .txt files
│   ├── processed/
│   │   └── finetune_pairs.jsonl
│   └── vectorstore/           # Persisted FAISS index
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/chatbot-rag
cd chatbot-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GROK_API_KEY (or use local LLM via GROK_BASE_URL)
```

### 3. Prepare data

```bash
# Layer 1: scrape Stripe docs → data/raw/stripe/
python scripts/scrape_stripe_docs.py

# Layer 2: download Bitext dataset → data/raw/bitext/ + data/processed/
python scripts/prepare_bitext_dataset.py

# Build vector index
python scripts/build_index.py
```

### 4. (Optional) Fine-tune

```bash
# Requires GPU for reasonable speed; CPU works but is slow
python -m app.models.finetune \
  --data data/processed/finetune_pairs.jsonl \
  --output data/finetuned_model \
  --epochs 2 \
  --max_samples 500

# Enable in .env:
# USE_FINETUNED_MODEL=true
# FINETUNED_MODEL_PATH=./data/finetuned_model
```

### 5. Run the API

```bash
python main.py
# or
uvicorn main:app --reload
```

API docs available at **http://localhost:8000/docs**

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 📡 API Reference

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I issue a refund on Stripe?"}'
```

**Response:**
```json
{
  "response": "To issue a refund on Stripe, navigate to the Payments section in your dashboard...",
  "sources": ["refunds.txt", "payments_overview.txt"]
}
```

**With chat history:**
```json
{
  "message": "What about partial refunds?",
  "history": [
    {"user": "How do I issue a refund?", "assistant": "Navigate to Payments..."}
  ]
}
```

**Streaming (SSE):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain webhooks", "stream": true}'
```

---

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@my_policy.pdf"
```

**Response:**
```json
{
  "message": "Document ingested successfully.",
  "filename": "my_policy.pdf",
  "chunks_added": 24
}
```

---

### `GET /search?q=...&k=4`

```bash
curl "http://localhost:8000/search?q=refund+policy&k=3"
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## 📊 Evaluation

```bash
python scripts/evaluate.py
```

Measures keyword hit rate and source retrieval accuracy across 6 test cases.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI LLM/embeddings |
| `LLM_MODEL` | `llama3-8b` | LLM model name |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `500` | Token chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `12` | Chunks retrieved per query |
| `USE_FINETUNED_MODEL` | `false` | Use fine-tuned model for generation |

---

## 🧠 Model Adaptation Details

### Option A — Fine-tuning (implemented)
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params, runs on CPU)
- Method: **LoRA** (Low-Rank Adaptation) via PEFT — only 0.1% of params trained
- Dataset: 500 stratified samples from Bitext (27 intents)
- Training: 2 epochs, batch size 4, lr 2e-4

### Option B — Structured Prompting (always active)
- System prompt enforces: answer only from context, say "I don't know" if not found
- Temperature 0.2 for factual consistency
- Chat history injection (last 3 turns)
