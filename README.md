# Otimização do modelo TRM (Tiny Recursive Model) para um problema que depende de memôria

This project is an implementation of the **Tiny Recursive Model (TRM)**, a highly efficient architecture adapted for memory-intensive problems.

Based on the paper "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871), the TRM uses a very small set of parameters (e.g., a simple 2-layer Transformer) in a **recursive loop**. 

Instead of relying on a massive number of parameters like a traditional Large Language Model (LLM), the TRM achieves computational depth by iteratively refining a latent "working memory" state. This approach makes it ideal for tasks where a solution must be built up or refined over multiple steps, using a persistent, evolving memory.


## Studied PAPERS

1. Less is More: Recursive Reasoning with Tiny Networks

https://arxiv.org/pdf/2510.04871

2. Hierarchical Reasoning Model

https://arxiv.org/pdf/2506.21734


## 🚀 Getting Started

Follow these steps to set up and run the project.

### 1. Install Dependencies

First, install the necessary Python libraries. You can use `uv` or standard `pip` with the provided `requirements.txt` file.

**Option A (Recommended): Using `uv`**
```bash
uv sync
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

**Running the code**
```bash
python run main.py
```

# Benchmarks

GoodAI – LTM Benchmark (GitHub e descrição do benchmark de memória de longo prazo).

https://github.com/GoodAI/goodai-ltm-benchmark


MemoryBench – Benchmark de Memória e Aprendizado Contínuo (links para dataset e código no GitHub; dataset no HuggingFace).

Dataset - https://huggingface.co/datasets/THUIR/MemoryBench

Benchmark - https://github.com/LittleDinoC/MemoryBench


RULER – Context Size Benchmark (resumo do objetivo e metodologia para contexto longo sintético).

https://github.com/NVIDIA/RULER