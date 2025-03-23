# LLaMA 2 Model from Scratch with PyTorch

## Overview
This project is an experiment to recreate the LLaMA 2 model architecture and inference code from scratch using PyTorch. The goal is to understand the core components of the model by implementing them manually and running inference using pre-trained weights.

## Model Components
The following components of the LLaMA 2 architecture have been implemented:

1. **RMS Layer**
2. **Grouped Multi-Query Attention (GQA)**
3. **Rotary Positional Embeddings (RoPE)**
4. **FeedForward Layer**
5. **Encoder Block**: Combines all LLaMA 2 architecture components
6. **Final Transformer Block**: Encapsulates all components and allows end-to-end input processing

## Tokenization
The project utilizes the **SentencePiece** library for tokenization. The inference code loads the 7B LLaMA 2 model parameters and generates tokens using **top-p sampling**.

## Inference Performance
The model was tested on an **Apple Mac Mini 2** with the following specifications:
- **Memory:** 8GB
- **CPU:** 8-core
- **GPU:** 10-core

Due to limited memory, the system ran out of RAM during inference, causing it to use **swap memory (20-30GB)** to move less frequently accessed memory pages from RAM to SSD. This introduced a significant bottleneck, resulting in an inference speed of **60-70 seconds per token**.

To optimize memory usage, PyTorch tensors were set to use **BFloat16Tensor** as the default type.

## Hyperparameters
The selected hyperparameters for inference are:

```json
{
  "dim": 4096,
  "multiple_of": 256,
  "n_heads": 32,
  "n_layers": 32,
  "norm_eps": 1e-05,
  "vocab_size": -1,
  "max_seq_len": 256,
  "batch_size": 1
}
```

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install torch sentencepiece
```

## Running the Model
To run inference with the model, execute the following script:

```bash
python inference.py
```
---