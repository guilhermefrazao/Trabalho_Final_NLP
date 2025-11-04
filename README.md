# Otimização do modelo TRM (Tiny Recursive Model) para um problema que depende de memôria

This project is an implementation of the **Tiny Recursive Model (TRM)**, a highly efficient architecture adapted for memory-intensive problems.

Based on the paper "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871), the TRM uses a very small set of parameters (e.g., a simple 2-layer Transformer) in a **recursive loop**. 

Instead of relying on a massive number of parameters like a traditional Large Language Model (LLM), the TRM achieves computational depth by iteratively refining a latent "working memory" state. This approach makes it ideal for tasks where a solution must be built up or refined over multiple steps, using a persistent, evolving memory.


## Studied PAPERS

1. Less is More: Recursive Reasoning with Tiny Networks

https://arxiv.org/pdf/2510.04871

2. Hierarchical Reasoning Model

https://arxiv.org/pdf/2506.21734

3. xLSTM: Extended Long Short-Term Memory

https://arxiv.org/pdf/2405.04517

## 🚀 Getting Started

Follow these steps to set up and run the project.

### 1. Install Dependencies

First, install the necessary Python libraries. You can use `uv` or standard `pip` with the provided `requirements.txt` file.

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
wandb login YOUR-LOGIN
```

**Windows**
```bash
pip install -U "triton-windows<3.4"  
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

**Running the code**

**Create the dataset**
```bash
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  
```

**Run Training**
```bash
python pretrain.py 
```

**Run Training with Custom Hyperparameters**
```bash
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_sudoku  ema=True global_batch_size=64
```

# Benchmarks

GoodAI – LTM Benchmark (GitHub e descrição do benchmark de memória de longo prazo).

Benchmark - https://github.com/GoodAI/goodai-ltm-benchmark


MemoryBench – Benchmark de Memória e Aprendizado Contínuo (links para dataset e código no GitHub; dataset no HuggingFace).

Dataset - https://huggingface.co/datasets/THUIR/MemoryBench

Benchmark - https://github.com/LittleDinoC/MemoryBench


RULER – Context Size Benchmark (resumo do objetivo e metodologia para contexto longo sintético).

Benchmark - https://github.com/NVIDIA/RULER