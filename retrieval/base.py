from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def get_context(self, query: str) -> list[str]:
        pass


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def get_embed_dim(self) -> int | None:
        pass
