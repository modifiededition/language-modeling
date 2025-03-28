from typing import Optional
import time
from model import Transformer, ModelArgs
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import torch
import json

from tqdm import tqdm
 
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
            checkpoint = torch.load(chk_path, map_location=device)
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
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        total_params_in_millions = total_params / 1_000_000
        print(f"Model has {total_params_in_millions:.2f}M parameters")
        
        if load_model:
            # deleting it not required to load, freqs are created on the run
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model,tokenizer,model_args) 

    def text_completion(self, 
                        prompts: list[str] ,# used to handle case where can pass multiples sentences in a single time
                        temperature: float = 0.6,
                        top_p: float = 0.9,
                        max_gen_len: Optional[int]=None,
                        device = "cpu"
                        ):
        
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        prompt_tokens =[ self.tokenizer.encode(prompt, out_type = int, add_bos = True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        max_prompt_len = max([len(prompt) for prompt in prompt_tokens  ])

        assert max_prompt_len <= self.args.max_seq_len
        
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)

        # create a list that will contain the generated tokens along with the prompt
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len ), pad_id, dtype=torch.long, device= device)

        for k,t in enumerate(prompt_tokens):
            # populate the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t,dtype=torch.long, device=device)

        prompt_tokens_mask = tokens!= pad_id# True if the token is prompt token else False 
        eos_reached = torch.tensor([False]*batch_size, device=device)

        for cur_pos in tqdm(range(1, total_len), desc = "Generating tokens"):
            with torch.no_grad():

                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            
            if temperature > 0:
                probs = torch.softmax(logits[:,-1]/temperature, dim = -1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # greedily select the token with the max prob.
                # here need to check, we might here also need to apply softmax befoer getting the max index
                next_token = torch.argmax(logits[:,-1], dim=-1)
            
            next_token = next_token.reshape(-1)

            # only replace token if its a padding token, since we dont care the output of the model on the prompt tokens
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # check EOS only for the output tokens not the the prompt tokens, since we dont care the output of the model on the prompt tokens even it gives EOS
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        
        out_tokens = [] # needed to remove eos ids
        out_text = []

        for current_prompt_tokens in tokens.tolist():
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):

        probs_sort, probs_index = torch.sort(probs,dim=-1,descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        # normalize the new probs
        probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_index, -1, next_token)
        return next_token

if __name__ == "__main__":
    torch.manual_seed(0)

    device =  "mps" if torch.mps.is_available() else "cpu"
    print(f"Device used: {device}")
    model  = LLaMA.build(
        checkpoint_path="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=64,
        max_batch_size=1,
        device=device
    )
    
    prompts = [
        "Newton first law "
    ]

    out_tokens, out_text = model.text_completion(prompts=prompts,max_gen_len=10,device=device)
    assert len(out_text) == len(prompts)

    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print("~"*50)
