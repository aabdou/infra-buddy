# AWS Docs RAG Assistant — Requirements

## Overview
A small RAG (Retrieval-Augmented Generation) assistant that answers natural-language questions about AWS services by retrieving relevant chunks from public AWS documentation and generating grounded answers via an LLM.

**Goal:** a scoped, finishable portfolio project demonstrating RAG pipeline construction, agent tool use, and evaluation — directly relevant to AWS infrastructure work.

---

## 1. Functional Requirements

### Core (must-have)
- [ ] Ingest a set of AWS documentation pages, scoped to 1–2 services (e.g. S3 + IAM)
- [ ] Chunk the docs into retrievable pieces
- [ ] Embed chunks and store them in a vector DB
- [ ] Accept a natural-language question (CLI input is fine)
- [ ] Retrieve the top-N relevant chunks for a given question
- [ ] Pass retrieved context + question to an LLM and return an answer
- [ ] Answer must cite/reference which doc chunk it came from

### Stretch (only after core works)
- [ ] One agent tool call — a function the LLM can invoke to check something dynamic (a mocked AWS API call is fine)
- [ ] Simple eval step: 10 test questions with expected answers, scored automatically
- [ ] Minimal web or Slack interface instead of CLI

---

## 2. Technical Requirements

| Component | Choice | Why |
|---|---|---|
| Language | Python | Best library support for this stack |
| Doc source | AWS docs (HTML or PDF) for 1–2 services | Scoped, real, publicly available |
| Chunking | Fixed-size or paragraph-based | Don't over-engineer this part |
| Embeddings | OpenAI/Anthropic embedding API, or local (sentence-transformers) | Either works; local avoids API cost |
| Vector store | ChromaDB | Reuse existing familiarity |
| LLM | Claude via API | Directly relevant |
| Interface | CLI to start | Don't build UI until logic works |

---

## 3. Non-Functional / Portfolio Requirements
- **Reproducible** — someone else can clone and run it with a README
- **Scoped** — 2 AWS services, ~20–50 doc chunks, not full AWS documentation
- **Documented decisions** — README section explaining chunking strategy, retrieval approach, and trade-offs made

---

## 4. Explicit Non-Goals
- No fine-tuning
- No multi-service coverage
- No production deployment/hosting
- No auth/user accounts