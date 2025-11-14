from models.mamba import generate_answer_mamba
from evaluate.ragas import evaluate_ragas
from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
from retrieval.models import HFEmbeddingModel, RerankerModel
from retrieval.naive import NaiveRetriever
from retrieval.reranker import RerankerRetriever

import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument("--reranker", action="store_true", help="Usa o reranker")
parser.add_argument("--naiverag", action="store_true", help="Usa o naive RAG")
parser.add_argument("--embeddings", action="store_true", help="Gerando os embeddings")

args = parser.parse_args()

def generate_embeddings(documents: str):
    emb_model = HFEmbeddingModel()
    embeddings_docs = emb_model.embed_text(documents)
    return embeddings_docs


def rerank_embeddings(query: str, documents: list[str]):
    reranker_model = RerankerModel()
    reranker_model.rank(query=query, documents=documents)


def dataset_PerLQTA():
    # load PerLT_Mem dataset
    dataset_mem = PerLTMem()

    dataset_qa = PerLTQA()

    character_names = dataset_mem.extract_character_names()

    random_character = random.randint(0, len(character_names) - 1)

    character_name = list(character_names)[random_character]

    character_data = dataset_qa.read_json_data('data/PerLTQA/Dataset/en/perltqa_en.json')

    character_facts = dataset_mem.read_json_data("data/PerLTQA/Dataset/en/perltmem_en.json")

    samples_Mem = dataset_mem.extract_sample(character_name)

    samples_QA = dataset_qa.extract_sample(character_name)
    
    return character_data, character_name, character_facts, random_character



if __name__ == "__main__":

    character_data, character_name, character_facts, random_character = dataset_PerLQTA()

    if args.embeddings:
        embeddings = generate_embeddings(character_data)

    else:
        faiss = "embeddings"
        embeddings = faiss

    prompt = character_data[character_name]

    if args.naiverag:
        rag = NaiveRetriever().get_context(prompt)

    elif args.reranker:
        rerank_embeddings(prompt, embeddings)
        rag = RerankerRetriever().get_context(prompt)

    prompt = prompt + rag

    answer = generate_answer_mamba(question=prompt)

    result = evaluate_ragas(questions=character_data[character_name], ground_truths=character_facts[random_character], contexts=character_facts[character_name], answers=answer)