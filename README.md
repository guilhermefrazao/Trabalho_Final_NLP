# Datasets 

Datasets utilizados para treinar modelos de Linguagem para tarefas relacionadas com a memória das LLMs.

1. **PerLTQA** 
Sobre - Dataset de QA focado em memória de longo prazo, que inclui memória semântica (Envolve fatos sobre o mundo e fatos pessoais/relacionamentos) e memória episódica (Histórico pessoal de dialogos e experiências),

Paper - https://arxiv.org/pdf/2402.16288
Repositório - https://github.com/Elvin-Yiming-Du/PerLTQA


2. **locomo**
Sobre - Dataset de conversas muito longo, é Multimodal e aprensenta várias conversas com diversos meses de diferença entre elas.

Paper - https://arxiv.org/pdf/2402.17753
Repositório - https://github.com/snap-research/LoCoMo

3. **LoCoGen**
Sobre - Datase para memória de Longo prazo.

Paper - https://aclanthology.org/2025.findings-acl.1014.pdf
Repositório - https://github.com/JamesLLMs/LoCoGen




# Models

Modelos treinados, otimizados para memória.

1. **Tiny Recursive Models**
2. **x-LSMT**
3. **Transformers-like**


# Evaluation

Benchmarks para conseguirmos avaliar os modelos desenvolvidos e treinados.

1. GoodAI – LTM Benchmark (GitHub e descrição do benchmark de memória de longo prazo).

Benchmark - https://github.com/GoodAI/goodai-ltm-benchmark


2. MemoryBench – Benchmark de Memória e Aprendizado Contínuo (links para dataset e código no GitHub; dataset no HuggingFace).

Benchmark - https://github.com/LittleDinoC/MemoryBench


3. RULER – Context Size Benchmark (resumo do objetivo e metodologia para contexto longo sintético).

Benchmark - https://github.com/NVIDIA/RULER


# Estrutura das pastas
```.
├── data    # Datasets para treino
│   ├── LoCoGen
│   ├── locomo 
│   └── PerLTQA
├── memory    # Memórias dos chats <Sujeito à mudança>
│   ├── memory_chat_1  # Exemplo de um chat
│   ├── faiss.index    # Banco vetorial
│   └── lookup.json    # Banco map
├── retrieval
│   ├── base.py      # Interface Retriever
│   ├── models.py    # Embedding e Reranker importados
│   ├── naive.py     # NaiveRetriever
│   ├── reranker.py  # RerankerRetriever
│   └── store.py     # Busca vetorial <Sujeito à mudança>
└── writing
    └── memory_writer.py    # Salva textos na memória <Place-holder>```