from typing import Optional
import time
from model import Transformer, ModelArgs
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import torch
import json
 
class LLaMA:
    def __init__(self, model:Transformer, tokenizer: SentencePieceProcessor , model_args:ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoint_path:str, tokenizer_path:str, load_model:bool, max_seq_len, max_batch_size, device):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_path).glob("*.pth"))
            assert len(checkpoints) > 0 , "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f"Loading Checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loading checkpoint in {time.time()-prev_time:.2f}s")
            prev_time  = time.time()

        with open(Path(checkpoint_path)/ "params.json", "r") as f:
            params = json.loads(f.read())

        model_args:ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # deleting it not required to load, freqs are created on the run
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model,tokenizer,model_args)    
    
if __name__ == "__main__":
    torch.manual_seed(0)

    device = "mps" if torch.mps.is_available() else "cpu"

    model  = LLaMA.build(
        checkpoint_path="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024//2,
        max_batch_size=1,
        device=device
    )
    print("Loaded Successfully!")
    
