import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from transformers_block import TransformerBlock

# --- Hierarchical Reasoning Model (HRM) ---
class HRM(nn.Module):
    """
    Implementação do Hierarchical Reasoning Model (HRM)  
    """
    def __init__(self, d_model, n_head, num_vocab, seq_len):
        super().__init__()
        self.d_model = d_model
        
        # 1. Embedding de Entrada (f_I)
        self.f_I_embedding = nn.Embedding(num_vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 2. Módulo de Baixo Nível (Rápido, f_L)
        self.f_L = TransformerBlock(d_model, n_head) 
        
        # 3. Módulo de Alto Nível (Lento, f_H)
        self.f_H = TransformerBlock(d_model, n_head)

        # 4. Cabeça de Saída (f_O)
        self.f_O = nn.Linear(d_model, num_vocab)

    def forward(self, x, N_cycles=4, T_steps=4):
        """
        x: Tokens de entrada (batch_size, seq_len)
        N_cycles: Número de ciclos lentos (N)
        T_steps: Número de passos rápidos por ciclo (T)
        
        """
        batch_size, seq_len = x.shape
        
        z_L = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        z_H = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)

        x_embed = self.f_I_embedding(x) + self.pos_embed
        
        for _ in range(N_cycles):
            for _ in range(T_steps):
                z_L = self.f_L(z_L + z_H + x_embed)
        
            z_H = self.f_H(z_H + z_L)


        output_logits = self.f_O(z_H)
        return output_logits 
