import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)# maps: indice -> embed vector

    def forward(self, x):
        return self.embedding(x)* math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len:int, dropout:float ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a zeros matrix of size seq_len, d_model
        pe = torch.zeros(seq_len,d_model)
        # create a vector of size seq_len,1
        position =  torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        # create a vector of size d_model/2
        div_term = torch.exp( torch.arange(0,d_model,2).float() * (-math.log(-10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # adding another dim for taking care of batch size

        pe = pe.unsqueeze(0) # 1,seq_len, d_model

        self.register_buffer("pe",pe)
    
    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class LayerNormization(nn.Module):
    
    def __init__(self, eps: float =10**-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative
        self.beta =  nn.Parameter(torch.zeros(1)) # additive

    
    def forward(self, x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha (x*mean) / (std + self.eps) + self.bias
    
    

