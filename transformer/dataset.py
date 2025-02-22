import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.seq_len = seq_len

        self.eos_token = torch.tensor(tokenizer_src.token_to_id(["[EOS]"]), dtype=torch.int64) 
        self.sos_token = torch.tensor(tokenizer_src.token_to_id(["[SOS]"]), dtype=torch.int64) 
        self.pad_token = torch.tensor(tokenizer_src.token_to_id(["[PAD]"]), dtype=torch.int64) 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):

        item = self.ds[index]

        src_text = item["translation"][self.src_lang]
        tgt_txt = item["translation"][self.tgt_lang]
        
        src_tokens  = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        src_pad_tokens_len = self.seq_len - len(src_tokens) - 2
        tgt_pad_tokens_len = self.seq_len - len(tgt_tokens) - 1

        if src_pad_tokens_len < 0 or tgt_pad_tokens_len <0:
            raise Exception("Seq length too small")
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens,dtype = torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*src_pad_tokens_len,dtype = torch.int64)
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token]*tgt_pad_tokens_len,dtype = torch.int64)
        ])

        labels = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*tgt_pad_tokens_len,dtype = torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert labels.size(0) == self.seq_len
        
        return {
            "encoder_input" : encoder_input, # (seq_len)
            "decoder_input" : decoder_input, # (seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask" : (decoder_input !=  self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "labels" :labels, # seq_len
            "src_txt": src_text,
            "tgt_txt": tgt_txt
         }
    

def causal_mask(size):

    # torch.triu will make the above diagonal values as 1
    mask = torch.triu(torch.ones((1, size,size)), diagonal=1).type(torch.int)
    return mask==0




