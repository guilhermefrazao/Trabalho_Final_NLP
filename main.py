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


#TODO: Adicionar a lógica dos outros datasets para realizar a inferência com outros tipos de memória
#TODO: Escalar a implementação da avaliação para o dataset, por enquanto somente avalia 1 pergunta e resposta por código rodado.
#TODO: Após realizar a avaliação, salvar os resultados dentro de algum arquivo, separado por modelo utilizado.
#TODO: Verificar a qualidade do RAG feito (Recupera documentos similares e relevantes?).
#TODO: Analisar se com base na arquitetura do modelo o resultado é satisfatório?

def find_rand(list: list):
    random_number = random.randint(0, len(list) - 1)
    chosen_data = list[random_number]
    return chosen_data


def generate_embeddings(documents: str, client):
    emb_model = HFEmbeddingModel()
    embeddings_docs = emb_model.embed_text(documents)
    MemoryRepository(client).add_memory(chat_id="PerLQTA_dataset", content=documents, category=0, embeddings=embeddings_docs)
    return embeddings_docs


def dataset_PerLQTA():
    # load PerLT_Mem dataset
    dataset_mem = PerLTMem()
    dataset_qa = PerLTQA()

    character_data = dataset_qa.read_json_data("data/PerLTQA/Dataset/en/perltqa_en.json")

    character_facts = dataset_mem.read_json_data("data/PerLTQA/Dataset/en/perltmem_en.json")

    character_names_mem = dataset_mem.extract_character_names()

    character_names_qa = dataset_qa.extract_character_names()

    character_name_mem = find_rand(list(character_names_mem))

    samples_Mem = dataset_mem.extract_sample(character_name_mem)

    try:
        samples_QA = dataset_qa.extract_sample(character_name_mem)

        question = find_rand(samples_QA["profile"])

        initial_prompt = question["Question"]
        
    except:
        character_name_qa = find_rand(list(character_names_qa))

        samples_QA = dataset_qa.extract_sample(character_name_qa)

        question = find_rand(samples_QA["profile"])

        initial_prompt = question["Question"]

        samples_Mem = ""

    return initial_prompt, samples_Mem, character_facts



if __name__ == "__main__":
    client = PersistentClient(path="memory/db")

    vector_store = ChromaVectorStore(
    client=client,
    embed_model=HFEmbeddingModel(),
    )

    answer = ""
    rag = ""

    #Foi criado somente o processamento com 1 dos datasets.
    initial_prompt, sample_mem, character_facts = dataset_PerLQTA()

    if args.embeddings:
        embeddings = generate_embeddings(json.dumps(character_facts), client)

    if args.naiverag:
        rag = NaiveRetriever(vector_store=vector_store, k=5).get_context(initial_prompt)

    elif args.reranker:
        rag = RerankerRetriever(vector_store, RerankerModel().model, 20, 5).get_context(initial_prompt)

    prompt = initial_prompt + rag[0]

    if args.mamba:
        answer = generate_answer_mamba(question=prompt)
    
    elif args.transformers:
        answer = generate_answer_transformers(question=prompt)

    elif args.xlstm:
        answer = generate_answer_xlstm(question=prompt)

    result = evaluate_ragas(questions=[initial_prompt], ground_truths=[sample_mem], contexts=[rag], answers=[answer])