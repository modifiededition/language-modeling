"""
This module contains the model class for the Llama2 model for using it in the inference process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:

    dim: int = 4096
    n_heads: int = 32 # Number of heads for queries
    n_kv_heads: Optional[int] = None # Number of heads for keys and values
    n_layers: int = 32
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multipler: Optional[float] = None
    norm_eps = 1e-6

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.tok_embedding = nn.Embedding(self.vlocab_size, self.dim)

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.output = nn.Linear(args.dim, self.vocab_size)

        self.freq_complex = precompute_theta_pos_frequencies(self.args.dim//args.n_heads, self.args.max_seq_len*2, device = self.args.device)
            
    def forward(self, tensor, start_pos):
        # tensor: (batch_size, seq_len)
        batch_size, seq_len = tensor.shape
        assert seq_len == 1, "Only one token at a time can be processed due to KV cache"

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        h = self.tok_embedding(tensor)

        # get rotatory position. Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos+seq_len] 
        freqs_complex  = self.freq_complex[start_pos:start_pos+seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
    



        








