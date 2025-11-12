from .base import Retriever
from .store import VectorStore
from .models import RerankerModel

class RerankerRetriever(Retriever):
    """Realiza busca vetorial + Reranker."""

    def __init__(self, 
                 vector_store: VectorStore, 
                 reranker_model: RerankerModel, 
                 initial_k: int, 
                 final_k: int):
        self.vector_store = vector_store
        self.reranker_model = reranker_model
        self.initial_k = initial_k
        self.final_k = final_k
        print("RerankerRetriever pronto")

    def get_context(self, query: str) -> list[str]:
        print("Executando get_context")

        initial_docs = self.vector_store.similarity_search(query, k=self.initial_k)  # Busca larga
        ranked_docs = self.reranker_model.rank(query, initial_docs)  # Rerank
        return ranked_docs[:self.final_k]  # Retorna os maiores ranks