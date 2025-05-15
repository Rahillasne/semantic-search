import json
import os
from typing import List

import faiss
import numpy as np
import typer
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
cli = typer.Typer()

# Pydantic model for query requests
type SearchRequest = BaseModel
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Globals populated at runtime
INDEX = None
METADATA: List[dict] = []
EMBED_MODEL: SentenceTransformer = None


@cli.command()
def index(data_path: str, index_path: str = "opportunity.index", metadata_path: str = "opportunity_meta.json"):
    """
    Ingest raw opportunity data, compute embeddings, build a FAISS index, and save index & metadata.

    
    Example:
      python semantic_search_pipeline.py index data/opps.jsonl
    """
    global EMBED_MODEL

    # 1. Load data
    if not os.path.exists(data_path):
        typer.echo(f"Data file not found: {data_path}")
        raise typer.Exit(code=1)

    with open(data_path, 'r') as f:
        records = [json.loads(line) for line in f]

    # 2. Initialize embedding model
    typer.echo("Loading embedding model...")
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Compute embeddings in batch
    typer.echo(f"Computing embeddings for {len(records)} items...")
    texts = [f"{r['title']}. {r['description']}" for r in records]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # 4. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 5. Save index and metadata
    typer.echo(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)

    typer.echo(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(records, f)

    typer.echo("Indexing complete!")


@cli.command()
def serve(index_path: str = "opportunity.index", metadata_path: str = "opportunity_meta.json", host: str = "0.0.0.0", port: int = 8000):
    """
    Start a FastAPI server to query the FAISS index for semantic search.

    Example:
      python semantic_search_pipeline.py serve
    """
    global INDEX, METADATA, EMBED_MODEL

    # Load metadata
    with open(metadata_path, 'r') as f:
        METADATA = json.load(f)

    # Load FAISS index
    INDEX = faiss.read_index(index_path)

    # Load embedding model
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    # Mount search endpoint
    @app.post("/search")
    def search(req: SearchRequest):
        """
        Perform semantic search over opportunities.
        """
        query_vec = EMBED_MODEL.encode([req.query], convert_to_numpy=True)
        distances, indices = INDEX.search(query_vec, req.top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            item = METADATA[idx]
            results.append({
                'id': item.get('id'),
                'title': item.get('title'),
                'description': item.get('description'),
                'score': float(dist)
            })
        return {'results': results}

    # Run the server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
