"""
This module contains the model class for the Llama2 model for using it in the inference process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

import math

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

def precompute_theta_pos_frequencies(head_dim : int, seq_len:int, device:str, theta_constant = 10000.0):

    # since euger formula can be gneralized to even number of dimensions
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    # create theta tensor (head_dim//2)
    theta_numerator = torch.arange(0,head_dim).float()
    theta = 1.0 / (theta_constant ** (theta_numerator / head_dim)).to(device)
    # create pos(m) tensor (seq_len)
    m = torch.arange(seq_len, device=device)

    # mutiple theta vector to each position(m) of vector sequence
    # (seq_len, head_dim//2)
    freqs = torch.outer(m, theta).float()

    # convert it into the complex form using: c = R* exp(i*m*theta), where R=1
    # c = exp(i*m*theta) = cos(m*theta) + i*sin(m*theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.tensor, freqs_complex: torch.tensor, device: str):
    # here freq_s_complex is tensor for the given position x
    # first we convert embedding of x into complex form
    # (batch_size, seq_len, h, head_dim) -> (batch_size, seq_len, h, head_dim//2,2)
    pair_wise_x = x.float().reshape(*x.shape[:-1],-1,2)
    # (batch_size, seq_len, h, head_dim//2,2) -> (batch_size, seq_len, h, head_dim//2)
    x_complex = torch.view_as_complex(pair_wise_x)
    #(seq_lem, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch_size, seq_len, h, head_dim//2) * (1, seq_len, 1, head_dim//2) -> (batch_size, seq_len, h, head_dim//2)
    rotated = x_complex * freqs_complex
    # convert back the complex form to real form
    # (batch_size, seq_len, h, head_dim//2) -> (batch_size, seq_len, h, head_dim//2, 2)
    x_out = torch.view_as_real(rotated)
    # flatten the last two dimensions
    # (batch_size, seq_len, h, head_dim//2, 2) -> (batch_size, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.tensor):
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 1) = (batch_size, seq_len, dim)
        # rsqrt = 1/sqrt
        return x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim = True) + self.eps)
    
    def forward(self, x: torch.tensor):
        # (dim) * (batch_size, seq_lem, dim) -> (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multipler:
            hidden_dim = int( args.ffn_dim_multipler * hidden_dim)
        # ROund the hidden dim to the nearest multiple of multiple_of parameter
        # example: if say hidden dim =7 and multiple_of = 5
        # ((hidden_dim + args.multiple_of - 1) // args.multiple_of) = (7 + 5 - 1) // 5 = 2
        # args.multiple_of  * 2 = 5*2=10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

def repeat_kv(x: torch.tensor, rep:int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        # (batch_size, seq_len, n_kv_heads, None, head_dim)
        x[:,:,:,None,:]
        .expand(batch_size, seq_len, n_kv_heads, rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()

        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        self.n_heads_q = args.n_heads
        
        # indicates how many times k and v heads to repeat to match the query heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim,  args.dim, bias = False)

        self.k_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim ))
        self.v_cache = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim ))
        
    def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.tensor):
        batch_size, seq_len, _  = x.shape
        # Since, seq_len is 1
        # (batch_size, 1, head_dim ) -> (batch_size, 1, num_heads_q * head_dim)
        xq = self.wq(x)
        # (batch_size, 1, head_dim ) -> (batch_size, 1, num_kv_heads * head_dim)
        xk = self.wk(x)
        # (batch_size, 1, head_dim ) -> (batch_size, 1, num_kv_heads * head_dim)
        xv = self.wv(x)

        # Split them among heads before applying attention

        # (batch_size, 1, num_heads_q * head_dim) -> (batch_size, 1, num_heads_q, head_dim) 
        xq = xq.view(batch_size,seq_len, self.n_heads_q, self.dim)
        # (batch_size, 1, num_kv_heads * head_dim) -> (batch_size, 1, num_kv_heads, head_dim) 
        xk = xk.view(batch_size,seq_len, self.n_kv_heads, self.dim)
        # (batch_size, 1, num_kv_heads * head_dim) -> (batch_size, 1, num_kv_heads, head_dim) 
        xv = xv.view(batch_size,seq_len, self.n_kv_heads, self.dim)

        # apply rototary positional encoding to query and keys
        # it does not change the shape of the tensors
        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        # replace the cache of keys and values with the latest token
        self.k_cache[:batch_size,start_pos: start_pos + seq_len] = xk
        self.v_cache[:batch_size,start_pos: start_pos + seq_len] = xv

        # retirive the keys and values from the cache till the given start pos
        # (batch_size, kv_seq_len, num_kv_heads, head_dim)
        keys = self.k_cache[:batch_size, 0:start_pos +seq_len] 
        values = self.v_cache[:batch_size, 0:start_pos + seq_len]

        # repeat heads for keys and values
        # (batch_size, kv_seq_len, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # prepare the tensors shape to calc attention
        # (batch_size, 1, num_heads_q, head_dim) - > (batch_size, num_heads_q, 1, head_dim)
        xq = xq.transpose(1,2)
        # (batch_size, seq_len_kv, num_heads_q, head_dim) - > (batch_size, num_heads_q, seq_len_kv, head_dim)
        keys = keys.transpose(1,2)
        # (batch_size, seq_len_kv, num_heads_q, head_dim) - > (batch_size, num_heads_q, seq_len_kv, head_dim)
        values = values.transpose(1,2)

        # calculate attention
        # (batch_size, num_heads_q, 1, head_dim) * (batch_size, num_heads_q, head_dim, kv_seq_len) ->  (batch_size, num_heads_q, 1, kv_seq_len)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(x)     

        # (batch_size,num_heads_q, 1, seq_len_kv) * (batch_size , num_heads_q, seq_len_kv, head_dim) - >  (batch_size , num_heads_q, 1, head_dim)
        out = torch.matmul(scores, values)

        # (batch_size, num_heads_q, 1, head_dim) -> (batch_size, 1, num_heads_q, head_dim) -> (batch_size, 1, dim )
        out = out.view(1,2).contiguous().view(batch_size,seq_len, -1)

        # (batch_size, 1, dim) -> (batch_size, 1, dim)
        return self.wo(out)
    
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # rms norm before attention
        self.attention_norm = RMSNorm(args.dim)

        # rms norm before feed forward
        self.ffn_norm = RMSNorm(args.dim)
    
    def forward(self, x: torch.tensor, start_pos:int, freqs_complex: torch.tensor):
        # here x/h is added first to make the skip connection
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) = (batch_size, seq_len, dim)
        h  = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
     
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
    