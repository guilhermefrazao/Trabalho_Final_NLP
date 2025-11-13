import logging
from typing import Any, Callable
from threading import Thread
from queue import Queue
from uuid import uuid4

from chromadb import PersistentClient
from chromadb.api import ClientAPI
from pydantic import BaseModel

from retrieval.base import EmbeddingModel
from retrieval.models import HFEmbeddingModel
from writing.predict import start_session, start_tokenizer, predict

_queue = Queue(maxsize=100)

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("memory")
logger.addHandler(logging.StreamHandler())


def _consumer(callback: Callable[[str, int, str], None]):
    """Function to consume the queue and run predictions."""
    start_session("writing/models/prediction/model.onnx")
    start_tokenizer("writing/models/prediction/tokenizer.json")
    while True:
        chat_id, input_data = _queue.get()
        category = predict(input_data)
        if category:
            print(f"\nMensagem {input_data} classificada como {category}", flush=True)
            callback(chat_id, category, input_data)


class Payload(BaseModel):
    chat_id: Any
    content: Any
    category: Any


class MemoryResult(BaseModel):
    payload: Payload
    score: float


class MemoryRepository:
    """
    Gerencia a escrita (adição) de novas memórias no banco de dados vetorial (índice FAISS) e no arquivo de lookup JSON.
    """

    def __init__(self, client: ClientAPI):
        self.client = client
        self.collection = self.client.get_or_create_collection("memories")

    def get_memories(
        self, chat_id: str, limit: int = 10, offset: int = 0
    ) -> list[Payload]:
        """
        Recupera todas as memórias de um chat_id.
        """
        r = self.collection.get(
            where={"chat_id": chat_id},
            include=["metadatas", "documents"],
            limit=limit,
            offset=offset,
        )
        if not r["metadatas"] or not r["ids"] or not r["metadatas"][0]:
            return []
        return [
            Payload(
                chat_id=r["metadatas"][i].get("chat_id"),
                content=r["documents"][i],
                category=r["metadatas"][i].get("category"),
            )
            for i in range(len(r["metadatas"])) if r["metadatas"] and r["documents"]
        ]

    def add_memory(
        self,
        chat_id: str,
        content: str,
        category: int,
        embeddings: list[float],
    ):
        """
        Adiciona nova memória ao chromadb.
        """
        payload = Payload(chat_id=chat_id, content=content, category=category)
        result = self.collection.add(
            embeddings=[embeddings],
            metadatas=[
                {
                    "chat_id": payload.chat_id,
                    "category": payload.category,
                }
            ],
            documents=[content],
            ids=[uuid4().hex],
        )
        print(f"Memória adicionada: {payload}", flush=True)
        print(f"Resultado: {result}", flush=True)

    def get_memory(
        self,
        chat_id: str,
        query: list[float],
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> list[MemoryResult]:
        """
        Recupera uma memória (query ou resposta) do banco de dados vetorial (índice FAISS) e do arquivo de lookup JSON.
        """
        r = self.collection.query(
            query_embeddings=[query],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
            where={"chat_id": chat_id},
        )
        results = []
        if (
            not r["metadatas"]
            or not r["metadatas"][0]
            or not r["metadatas"][0][0]
            or not r["distances"]
            or not r["distances"][0]
            or not r["distances"][0][0]
            or not r["documents"]
            or not r["documents"][0]
            or not r["documents"][0][0]
            or not r["ids"]
        ):
            return results
        for i in range(min(len(r["metadatas"][0]), top_k)):
            if r["distances"][0][i] > max_distance:
                break
            results.append(
                MemoryResult(
                    payload=Payload(
                        chat_id=r["metadatas"][0][i].get("chat_id"),
                        content=r["documents"][0][i],
                        category=r["metadatas"][0][i].get("category"),
                    ),
                    score=r["distances"][0][i],
                )
            )
        return results


class Memory:
    """
    Gerencia a escrita (adição) de novas memórias no banco de dados vetorial (índice FAISS) e no arquivo de lookup JSON.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        db_path: str,
    ):
        client = PersistentClient(path=db_path)
        self.db = MemoryRepository(client)

        self.embed_model = embedding_model

        self.dimension = self.embed_model.get_embed_dim()

        self.consumer_thread = Thread(target=_consumer, args=(self._save_to_disk,))
        self.consumer_thread.start()

    def _save_to_disk(self, chat_id: str, category: int, content: str):
        """
        Adiciona uma nova memória (query ou resposta) EM MEMÓRIA e salva no disco.
        """
        print(f"\nSalvando memória {content} como {category}", flush=True)
        embeddings = self.embed_model.embed_text(content)
        self.db.add_memory(chat_id, content, category, embeddings)
        print("Memória salva", flush=True)

    def add_memory(self, chat_id: str, content: str):
        """
        Adiciona uma nova memória (query ou resposta) EM MEMÓRIA.
        Esta função NÃO salva no disco. (seria muito pesado salvar no disco toda hora, é melhor adicionar várias memórias em um "cache" e carregar no disco periodicamente)
        """
        _queue.put((chat_id, content))

    def get_memory(self, chat_id: str, query: str) -> list[MemoryResult]:
        """
        Recupera uma memória (query ou resposta) do banco de dados vetorial (índice FAISS) e do arquivo de lookup JSON.
        """
        embeddings = self.embed_model.embed_text(query)
        return self.db.get_memory(chat_id, embeddings)


if __name__ == "__main__":
    m = Memory(HFEmbeddingModel(), "memory/db")
    chat_id = input("Digite o chat_id (pode ser seu nome, tanto faz): ")
    while True:
        opt = input(
            "Digite 1 para adicionar uma memória, 2 para recuperar uma memória, 3 para sair: "
        )
        if opt == "3":
            break
        elif opt == "1":
            q = input("Digite uma pergunta: ")
            m.add_memory(chat_id, q)
            print(m.get_memory(chat_id, q))
            continue
        elif opt == "2":
            print(m.db.get_memories(chat_id))
            continue
