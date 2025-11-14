from transformers import MambaForCausalLM, AutoTokenizer

def generate_answer_mamba(question : str):
    print(f"Question: {question}")

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")

    input_ids = tokenizer(question, return_tensors="pt")["input_ids"]

    output = model.generate(input_ids.to(model.device), max_new_tokens=20)
    print(tokenizer.batch_decode(output))

    return tokenizer.batch_decode(output)
