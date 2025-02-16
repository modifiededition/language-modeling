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
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormization()
    
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.layer_norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_layer: MultiHeadAttention, feed_forward_layer:FeedForwardBlock, dropout:float ):
        super().__init__()

        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_layers = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # src_mask required to avoid interaction of padding tokens during attn
    def forward(self, x, src_mask):

        x = self.residual_layers[0](x, lambda x: self.self_attention_layer(x,x,x,src_mask))
        x = self.residual_layers[1](x, self.feed_forward_layer)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormization()
    
    def forward(self,x, mask):
        for layer in self.layers:
            x  = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout:float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_layers = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self,x, encoder_output, src_mask, tgt_mask):

        x = self.residual_layers[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_layers[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output, src_mask))
        x = self.residual_layers[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.norm = LayerNormization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int, vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder: Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding,tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):

        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encode(src,src_mask)

    def decoder(self, encoder_output, src_mask, tgt, tgt_mask):

        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output,src_mask,tgt,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)
    

## Building Transformer based on the given hyper-paramters

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model =512, N =6, h=8, d_ff = 2048,dropout=0.1):

    src_embed = InputEmbedding(d_model,src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos_layer = PositionalEncoding(d_model,src_seq_len)
    tgt_pos_layer = PositionalEncoding(d_model, tgt_seq_len)

    encoder_blocks = []

    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attention_block, feed_forward_block, dropout))
    
    decoder_blocks = []

    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model,h,dropout)
        cross_attention_block = MultiHeadAttention(d_model,h,dropout)
        
        feed_forward_block = FeedForwardBlock(d_model,d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attention_block,cross_attention_block,feed_forward_block,dropout))

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer =  Transformer(encoder,decoder,src_embed,tgt_embed,src_pos_layer,tgt_pos_layer, proj_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
