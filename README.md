# Berra's Transformer-based AI

A GPT-style transformer built from scratch in PyTorch, pretrained on WikiText-2, then finetuned on my own iMessage data to generate text in my conversational style.

## Pipeline

1. **`data_cleaning.py`** — downloads and cleans WikiText-2 (lowercasing, removing headers/artifacts, fixing spacing), saves train/test splits to disk.
2. **`imessage_cleaning.py`** — parses raw exported iMessage chat logs into clean `ME:` / `THEM:` labeled text, filtering out spam, attachments, and reactions.
3. **`model.py`** — defines the `Transformer` architecture: token + positional embeddings, a pre-norm `TransformerEncoder` stack with causal masking, and a linear output head.
4. **`train.py`** — pretrains the model on cleaned WikiText-2, using next-token prediction over flattened token streams. Logs to Weights & Biases and saves checkpoints.
5. **`finetune.py`** — continues training from a pretraining checkpoint on `THEM:`/`ME:` message pairs, masking the loss so the model only learns to predict my responses, not the context.
6. **`generate.py`** — loads a finetuned checkpoint and generates text autoregressively, with temperature, top-p (nucleus) sampling, and a repetition penalty.

## Usage

```bash
# 1. Clean pretraining data
python data_cleaning.py

# 2. Parse personal iMessage data (expects raw exports in data/raw/chats_new)
python imessage_cleaning.py

# 3. Pretrain on WikiText-2
python train.py

# 4. Finetune on iMessage data
python finetune.py

# 5. Generate text
python generate.py
```

## Notes

- Uses the GPT-2 tokenizer (byte-level BPE) via the `tokenizers` library.
- Model hyperparameters (`d_model`, `seq_len`, `nhead`, `n_layers`) must match between training and generation scripts.
- Runs on Apple Silicon (`mps`) with CPU fallback.
- This is a from-scratch learning project — not a wrapper around a pretrained model.

## Status / known limitations

- Small model size + limited pretraining data means generation quality is still rough (locally fluent, not always globally coherent).
- Actively debugging generation stopping behavior and padding handling.
