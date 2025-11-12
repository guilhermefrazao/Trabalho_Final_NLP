from .models import EmbeddingModel
import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel, index_path: str, lookup_path: str):
        self.embedding_model = embedding_model
        self.index = faiss.read_index(index_path)
        with open(lookup_path, 'r', encoding='utf-8') as f:
            self.document_lookup = json.load(f)
            
    
    def similarity_search(self, query: str, k: int) -> list[str]:
        """Encontra K documentos similares a partir de uma query."""

        # Embeda a query
        query_vector = self.embedding_model.embed_text(query)
        query_vector_np = np.array([query_vector]).astype('float32')  # FAISS espera um array numpy 2D float32

        # Busca no índice
        D_scores, I_indices = self.index.search(query_vector_np, k)  # D = Distâncias (scores), I = Índices (os IDs)
        found_ids = I_indices[0]  # Só a primeira lista, pois é só 1 query

        # Retorna documentos
        results = []
        for doc_id in found_ids:
            str_id = str(doc_id)  # chaves do JSON são strings.
            
            if str_id in self.document_lookup:
                results.append(self.document_lookup[str_id])
            else:
                print(f"ERRO: ID {doc_id} encontrado no FAISS, mas não no lookup.json")
        
        print(f"Encontrados {len(results)} documentos")
        return results