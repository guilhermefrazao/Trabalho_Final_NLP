from models.mamba import generate_answer_mamba
from models.transformers import generate_answer_transformers
from models.xlstm import generate_answer_xlstm
from writing.memory import MemoryRepository
from retrieval.store import ChromaVectorStore
from evaluate.ragas import evaluate_ragas
from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
from retrieval.models import HFEmbeddingModel, RerankerModel
from retrieval.naive import NaiveRetriever
from retrieval.reranker import RerankerRetriever

import argparse
import json
import random
from chromadb import PersistentClient

parser = argparse.ArgumentParser()

parser.add_argument("--reranker", action="store_true", help="Usa o reranker")
parser.add_argument("--naiverag", action="store_true", help="Usa o naive RAG")
parser.add_argument("--embeddings", action="store_true", help="Gerando os embeddings")
parser.add_argument("--mamba", action="store_true", help="Chamando o modelo mamba")
parser.add_argument("--transformers", action="store_true", help="Chamando o modelo de trasnformers")
parser.add_argument("--xlstm", action="store_true", help="Chamando o modelo xlstm")

args = parser.parse_args()

def generate_embeddings(documents: str, client):
    emb_model = HFEmbeddingModel()
    embeddings_docs = emb_model.embed_text(documents)
    MemoryRepository(client).add_memory(chat_id="PerLQTA_dataset", content=documents, category=0, embeddings=embeddings_docs)
    return embeddings_docs


def dataset_PerLQTA():
    # load PerLT_Mem dataset
    dataset_mem = PerLTMem()

    dataset_qa = PerLTQA()
    
    character_data = dataset_qa.read_json_data('data/PerLTQA/Dataset/en/perltqa_en.json')

    character_facts = dataset_mem.read_json_data("data/PerLTQA/Dataset/en/perltmem_en.json")

    character_names = dataset_mem.extract_character_names()

    random_character = random.randint(0, len(character_names) - 1)

    character_name = list(character_names)[random_character]

    samples_Mem = dataset_mem.extract_sample(character_name)

    samples_QA = dataset_qa.extract_sample(character_name)
    
    return samples_QA, samples_Mem, character_facts



if __name__ == "__main__":
    client = PersistentClient(path="memory/db")

    #Foi criado somente o processamento com 1 dos datasets.
    sample_qa, sample_mem, character_facts = dataset_PerLQTA()

    if args.embeddings:
        embeddings = generate_embeddings(json.dumps(character_facts), client)

    vector_store = ChromaVectorStore(
    client=client,
    embed_model=HFEmbeddingModel(),
    )

    prompt = sample_qa

    if args.naiverag:
        rag = NaiveRetriever(vector_store=vector_store, k=5).get_context(prompt)

    elif args.reranker:
        rag = RerankerRetriever(vector_store, RerankerModel().model, 20, 5).get_context(prompt)

    prompt = prompt + rag

    if args.mamba:
        answer = generate_answer_mamba(question=prompt)
    
    elif args.transformers:
        answer = generate_answer_transformers(question=prompt)

    elif args.xlstm:
        answer = generate_answer_xlstm(question=prompt)

    result = evaluate_ragas(questions=sample_qa, ground_truths=sample_mem, contexts=sample_mem, answers=answer)