# Retrieval-Augmented Generation for WWI History

This project is a **Retrieval-Augmented Generation (RAG)** system that answers historical questions about **World War I** using a **fully local stack**. It combines:

- **Paragraph-level sparse retrieval** using **SPLADE**
- **Sentence-level reranking** with a **cross-encoder**
- **Local language generation** with **Ollama** (e.g., llama3:8b)
- A corpus of ~1,200 Wikipedia articles on WWI, automatically chunked and downloaded from HuggingFace

No API keys required. Everything runs locally.


## Quick Start

### 1. Set up environment

```bash
conda create -n wwi python=3.12
conda activate wwi
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install and pull a model via Ollama

```
pip install ollama
ollama pull llama3.1:8b

```

### 3. Ask a question
```
python src/answer_engine.py --q "What happened in 1915?"
```
Optional arguments

```
--retriever splade            # or 'openai-embed' if using OpenAI (disabled by default)
--model llama3:8b             # Ollama model to use
--reranker_topk 30            # Number of top-ranked sentences to keep

```

## Evaluations Results
The system is evaluated against base LLM generation (no retrieval) to measure the benefit of retrieval + reranking.


### Evaluation Results

| Retriever        | Reranker         | Score (%) |
|------------------|------------------|-----------|
| None             | None             | 40.5      |
| Splade           | CrossEncoder     | 43.5      |
| Splade           | OpenAI Embedding | 62.0      |
| OpenAI Embedding | OpenAI Embedding | 52.5      |
