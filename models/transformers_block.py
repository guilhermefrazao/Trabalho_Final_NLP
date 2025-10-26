import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# --- Transformer BLock ---
class TransformerBlock(nn.Module):
    """ Um bloco Transformer simples (Self-Attention + MLP) """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)

        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)

        x = self.norm2(x + ffn_out)

        return x

