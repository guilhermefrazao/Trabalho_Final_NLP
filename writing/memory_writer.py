from retrieval.models import EmbeddingModel
import faiss
import numpy as np
import json
import os

######################### ARQUIVO PLACE-HOLDER
######################### ARQUIVO PLACE-HOLDER
######################### ARQUIVO PLACE-HOLDER
######################### ARQUIVO PLACE-HOLDER
######################### ARQUIVO PLACE-HOLDER

class MemoryWriter:
    """
    Gerencia a escrita (adição) de novas memórias no banco de dados vetorial (índice FAISS) e no arquivo de lookup JSON.
    
    Esta classe opera em memória e só salva no disco quando o método .save_to_disk() é chamado.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, index_path: str, lookup_path: str):
        self.embed_model = embedding_model
        self.index_path = index_path
        self.lookup_path = lookup_path
        
        # Pega a dimensão do vetor do modelo (Necessário se precisarmos criar um índice do zero)
        self.dimension = self.embed_model.model.get_sentence_embedding_dimension()
        
        self.lookup = self._load_lookup()
        self.index = self._load_index()


    def _load_lookup(self) -> dict:
        """
        procura um arquivo lookup.json para o chat atual, se não encontrar ele cria
        """


    def _load_index(self):
        """
        procura um arquivo faiss.index para o chat atual, se não encontrar ele cria
        """


    def add_memory(self, chat_id: str, category: str, content: str):
        """
        Adiciona uma nova memória (query ou resposta) EM MEMÓRIA.
        Esta função NÃO salva no disco. (seria muito pesado salvar no disco toda hora, é melhor adicionar várias memórias em um "cache" e carregar no disco periodicamente)
        """


    def save_to_disk(self):
        """
        Salva o índice FAISS e o lookup JSON (em memória) de volta no disco.
        """