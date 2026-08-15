---
title: Multimodal Document AI RAG
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: src/ui/streamlit_app.py
pinned: false
---
# Multimodal Document Intelligence & RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![CLIP](https://img.shields.io/badge/Model-CLIP--ViT--B--32-orange)

An end-to-end RAG (Retrieval-Augmented Generation) system for parsing, indexing, and searching complex PDF documents using Multimodal Embeddings (CLIP) and Qdrant vector database.

## 📐 Architecture Overview

1. **Document Ingestion:** PDF documents are processed page-by-page, rendering both high-resolution page images and raw text.
2. **Multimodal Encoding:** Text and visual features are encoded into a shared vector space via CLIP embeddings (`clip-ViT-B-32`).
3. **Vector Indexing:** Embeddings along with payload metadata (page number, filename, raw text snippet) are indexed in Qdrant Vector DB.
4. **Search & Retrieval:** Semantic search endpoint maps user text queries against multimodal document representations.

## 🚀 Quick Start (Docker)

### 1. Run Vector DB
```bash
docker-compose up -d
