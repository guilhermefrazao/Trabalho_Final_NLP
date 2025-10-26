import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from transformers_block import TransformerBlock

# --- Tiny Recursive Model (TRM) ---
class TRM(nn.Module):
    """
    Implementação do Tiny Recursive Model (TRM)
    """
    def __init__(self, d_model, n_head, num_vocab, seq_len, 
                 n_sup_steps=16, n_reasoning_steps=1):
        super().__init__()
        
        self.n_sup_steps = n_sup_steps           
        self.n_reasoning_steps = n_reasoning_steps 
        
        # Embeddings para Input (x), Prediction (y)
        self.x_embedding = nn.Embedding(num_vocab, d_model)
        self.y_embedding = nn.Embedding(num_vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Estado Latente (z) 
        self.z_init = nn.Parameter(torch.randn(1, seq_len, d_model))

        # A rede recursiva única (ex: 2 camadas) 
        self.net = nn.Sequential(
            TransformerBlock(d_model, n_head),
            TransformerBlock(d_model, n_head)
        )
        
        # Cabeça de Saída (Reverse Embedding) 
        self.output_head = nn.Linear(d_model, num_vocab)

    def forward(self, x, y):
        """
        x: A "pergunta" (Input x)
        y: Uma "resposta" inicial (Prediction y) 
        """
        batch_size, seq_len = x.shape
        
        x = self.x_embedding(x) + self.pos_embed
        y = self.y_embedding(y) + self.pos_embed
        
        z = self.z_init.expand(batch_size, -1, -1)

        for _ in range(self.n_sup_steps): 
            z_temp = z
            for _ in range(self.n_reasoning_steps):
                z_temp = self.net(x + y + z_temp)
            z = z_temp

            y = self.net(x + y + z)

        output_logits = self.output_head(y)
        return output_logits 
    