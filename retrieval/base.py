from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def get_context(self, query: str) -> list[str]:
        pass

