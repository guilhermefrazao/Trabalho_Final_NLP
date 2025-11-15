from .models import EmbeddingModel
from chromadb import PersistentClient
import numpy as np
import json

class ChromaVectorStore:
    def __init__(self, client: PersistentClient, embed_model):
        self.collection = client.get_or_create_collection("memories")
        self.embed_model = embed_model

    def similarity_search(self, query, k):
        emb = self.embed_model.embed_text(query)
        result = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents"]
        )
        return result["documents"][0]