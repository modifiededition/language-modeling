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
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, dff:int, dropout:float):
        super().__init__()

        self.linear1 = nn.Linear(d_model,dff) # w1 and b1
        self.linear2 = nn.Linear(dff, d_model) # w2 and b2

        self.dropout  = self.dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model%h == 0, "d_model is not divisible by h"

        self.d_k = self.d_model // self.h


        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)
        self.wo = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask, dropout:nn.Dropout):

        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_score = (query @ key.transporse(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill(mask == 0, -1e9)
        
        attention_score = attention_score.softmax(dim=-1) #(batch_size, h, seq_len, seq_len)

        attention_values = attention_score @ value

        return attention_values, attention_score
    
    def forward(self, q, k, v, mask):

        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shap[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shap[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value =  value.view(value.shap[0],value.shape[1],self.h, self.d_k).transpose(1,2)
        
        x, self.attention_score = MultiHeadAttention.attention(query,key,value,mask, self.dropout)
        
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k)

        return self.wo(x)



        
        