from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
    answer_relevancy,
)


def evaluate_ragas(questions, ground_truths, contexts, answers, title=""):
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "contexts": contexts,
        "answer": answers
    })
    
    print(f"\n==== Avaliando: {title} ====\n")
    
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
            answer_relevancy,
        ]
    )
    
    print(result)
    return result
