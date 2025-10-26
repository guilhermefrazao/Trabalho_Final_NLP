import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


from models.hrm import HRM
from models.trm import TRM


# TODO: Encontrar um dataset mais visual para testar a capacidade dos modelos
# TODO: Encontrar um benchmark de memôria para avaliar a capacidade da arquitetura que iremos propor
# TODO: Adicionar tudo isso dentro do Readme.md


if __name__ == '__main__':
    D_MODEL = 64
    N_HEAD = 4
    SEQ_LEN = 32
    VOCAB_SIZE = 1000
    BATCH_SIZE = 8


    # --- Testando HRM ---
    print("Testando HRM...")
    hrm_model = HRM(d_model=D_MODEL, n_head=N_HEAD, num_vocab=VOCAB_SIZE, seq_len=SEQ_LEN)

    input_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    output_hrm = hrm_model(input_x, N_cycles=4, T_steps=4)

    total_params = sum(p.numel() for p in hrm_model.parameters())

    print(f"Total number of parameters: {total_params:,}")
    print("Entrada HRM shape:", input_x.shape)
    print("Saída HRM shape:", output_hrm.shape)
    print("-" * 30)


    # --- Testando TRM ---
    print("Testando TRM...")
    trm_model = TRM(d_model=D_MODEL, n_head=N_HEAD, num_vocab=VOCAB_SIZE, seq_len=SEQ_LEN, 
                  n_sup_steps=16, n_reasoning_steps=1)
    
    input_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    input_y = torch.zeros_like(input_x) 

    output_trm = trm_model(input_x, input_y)

    total_params = sum(p.numel() for p in trm_model.parameters())
    
    print("Entrada TRM (x) shape:", input_x.shape)
    print("Entrada TRM (y_init) shape:", input_y.shape)
    print("Saída TRM shape:", output_trm.shape)