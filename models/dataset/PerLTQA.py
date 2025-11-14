from data.PerLTQA.Dataset.dataset import PerLTMem, PerLTQA
import random

def create_question_from_data():
    dataset_mem = PerLTMem()

    dataset_qa = PerLTQA()

    character_names = dataset_mem.extract_character_names()

    character_name = random.choice(list(character_names))

    character_data = dataset_qa.read_json_data('data/PerLTQA/Dataset/en/perltqa_en.json')

    #samples_Mem = dataset_mem.extract_sample(character_name)

    #samples_QA = dataset_qa.extract_sample(character_name)

    return character_data[character_name]