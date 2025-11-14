from models.mamba import generate_answer_mamba
from evaluate.ragas import evaluate_ragas
from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
import random


if __name__ == "__main__":
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

    #rag_response = 

    prompt = character_data[character_name]

    answer = generate_answer_mamba(question=prompt)

    result = evaluate_ragas(questions=character_data[character_name], ground_truths=character_facts[random_character], contexts=character_facts[character_name], answers=answer)