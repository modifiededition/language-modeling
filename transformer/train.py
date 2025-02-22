import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import random_split,DataLoader

from pathlib import Path

from dataset import BilingualDataset

from model import build_transformer

from config import get_config, get_weights_file_path

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(model=WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds,lang):
    for item in ds:
        yield item["translation"][lang]

def get_ds(config):
    ds_raw = load_dataset(path='opus_books', data_dir= f"{config['lang_src']}-{config['lang_tgt']}")["train"]

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])

    # split ds for train/test
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size  = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_lan_src = 0
    max_lan_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]])
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]])
        max_lan_src = max(max_lan_src , len(src_ids))
        max_lan_tgt = max(max_lan_tgt, len(tgt_ids))
    
    print(f"Max length of the src sentence: {max_lan_src}")
    print(f"Max length of the tgt sentence: {max_lan_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size= config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):

    return build_transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
    src_seq_len= config["seq_len"], tgt_seq_len= config["seq_len"],N = config["num_layers"],
    d_ff=config["d_ff"],h=config["num_heads"],dropout=config["dropout"],
    d_model=config["d_model"])

# training loop

def train_model(config):

    # define the device
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    print("Using Device: ", device)

    # create the weights folder
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # get the data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt =get_ds(config=config)
    # get the trasnformer model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # tensorboard
    writer = SummaryWriter(config["experiment_name"])

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_file_path =  get_weights_file_path(config, config=["preload"])
        print("Preloading Model: ", model_file_path)
        state = torch.load(model_file_path,map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    # define the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epooch {epoch :02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1,1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
            labels = batch["labels"].to(device) # (batch_size, seq_len)

            with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
                encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
                logits = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

                # logits.view(-1, tokenizer_tgt.get_vocab_size() -> (batch_size * seq_len, vocab_size)
                # labels.view(-1) -> (batch_size * seq_len)

                loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), labels.view(-1))
            
            #log loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if config["debug"] and global_step > 10:
                break

        # save the model
        model_file_path = get_weights_file_path(config, str(epoch))

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict()
        }
        torch.save(state, model_file_path)

        if config["debug"]:
            break

if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    config = get_config()
    train_model(config)




    
    

