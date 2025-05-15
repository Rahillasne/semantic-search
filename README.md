# Semantic Search Pipeline

A self-contained semantic search service powered by FAISS and SBERT.

## Features
- **Indexing**: Ingest and embed raw opportunity data.
- **Vector Search**: High-performance nearest-neighbor search with FAISS.
- **Serving**: Production-ready REST API via FastAPI.
- **CLI**: Easy commands to index & serve using [Typer](https://typer.tiangolo.com/).

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
