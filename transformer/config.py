from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "num_layers": 3,
        "num_heads": 8,
        "d_ff": 2048 // 2,
        "dropout": 0.1,
        "debug": False,
        "run_validation_nums": 500,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = model_basename + epoch + ".pt"
    return str(Path(".") / model_folder / model_filename)
