import torch
from model import Transformer
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

d_model, seq_len, nhead, n_layers = 256, 128, 4, 6

# using gpt2 tokenizer which uses byte level encoding and byte pair encoding
tokenizer = Tokenizer.from_pretrained("gpt2")
# knowing vocab size is important for the model. tokenizer has get_vocab_size for it.
vocab_size = tokenizer.get_vocab_size()

# rebuild the base model + same LoRA config used in finetune.py
model = Transformer(vocab_size, d_model, seq_len, nhead, n_layers).to(device)
model.load_state_dict(torch.load('checkpoints/best.pt', map_location=device))

config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj", "linear"]
)
model = get_peft_model(model, config)

# now load your fine-tuned (LoRA) checkpoint into this PEFT-wrapped model
model.load_state_dict(torch.load('checkpoints_ft/best.pt', map_location=device))

# merge LoRA weights into the base weights, then discard the adapter wrapper
merged_model = model.merge_and_unload()

# save as a plain state dict — same format as your original pretrained checkpoint
torch.save(merged_model.state_dict(), 'checkpoints_ft/merged_best.pt')