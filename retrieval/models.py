from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieval.base import EmbeddingModel

class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = 'google/embeddinggemma-300m'):
        self.model = SentenceTransformer(model_name)
        print(f"-> EmbeddingModel carregado: {model_name}")

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def get_embed_dim(self) -> int | None:
        return self.model.get_sentence_embedding_dimension()
    

class RerankerModel:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L6-v2'):
        self.model = CrossEncoder(model_name)
        print(f"-> RerankerModel carregado: {model_name}")

    def rank(self, query: str, documents: list[str]) -> list[str]:
        pairs = [(query, doc) for doc in documents]  # Cria pares (query, doc) para o CrossEncoder

        scores = self.model.predict(pairs)  # Obtém os scores

        ranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]  # Ordena os documentos pelos scores
        return ranked_docs
