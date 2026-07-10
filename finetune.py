# imports
import torch
from torch import nn
from tokenizers import Tokenizer
import wandb
import os
from model import Transformer
from peft import LoraConfig, get_peft_model

# for reproducability
torch.manual_seed(873947)

# use gpu instead of cpu 
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# using gpt2 tokenizer which uses byte level encoding and byte pair encoding
tokenizer = Tokenizer.from_pretrained("gpt2")
# knowing vocab size is important for the model. tokenizer has get_vocab_size for it.
vocab_size = tokenizer.get_vocab_size()

# using my imessage data. open it as 'r'. with closes it when done. the encoding handles special characters
with open('data/raw/imessage_parsed.txt', 'r', encoding='utf-8') as f:
    # removes leading and trailing whitespaces and skips a line if it is empty after stripping
    lines = [l.strip() for l in f.readlines() if l.strip()]

# how many previous tokens (texts) to look at as context
CONTEXT_WINDOW = 4

pairs = []
# loops through every line
for i in range(len(lines) - 1):
    # only creates a pair if the current message starts with THEM: and the next one starts with ME:
    if lines[i].startswith('THEM:') and lines[i+1].startswith('ME:'):
        # goes back 4 messages for context. does max to make sure we don't go into negative messages
        context_start = max(0, i - CONTEXT_WINDOW + 1)
        # joins all messages in the context with a new line
        context = '\n'.join(lines[context_start:i+1])
        # records the ME: line as my response, which is what the model will learn to predict
        response = lines[i+1]
        # adds the context and response to pairs
        pairs.append((context, response))

print(f"Found {len(pairs)} pairs")

# hyperparameters for depth and size of model etc
d_model = 256
seq_len = 128
nhead = 4
n_layers = 6
batch_size = 16
lr = 2e-4
n_steps = 5000
eval_every = 200
# tells model to ignore the ME: messages? loss masking? 
IGNORE_INDEX = -100

def build_sample(context, response):
    # tokenizes the context (prompt) with an added newline
    context_ids = tokenizer.encode(context + '\n').ids
    # tokenize the response
    response_ids = tokenizer.encode(response + '\n').ids
    # concatenate into one flat sequence
    ids = context_ids + response_ids
    # for every context token, it puts ignore index so the loss function
    # skips it so the model only learns to predict the responses
    mask = [IGNORE_INDEX] * len(context_ids) + response_ids

    # truncates ids and mask ?? kinda confused
    # makes them long enough to shift to get right number? 
    ids = ids[:seq_len + 1]
    mask = mask[:seq_len + 1]

    # if the  sequence is too short, pad it
    if len(ids) < seq_len + 1:
        pad = seq_len + 1 - len(ids)
        ids = ids + [0] * pad
        # pads with ignore index so doesn't affect loss
        mask = mask + [IGNORE_INDEX] * pad

    return ids, mask

# goes through all of the context-response pairs, puts them through build_sample, stacks results into lists
all_ids = []
all_masks = []
for context, response in pairs:
    ids, mask = build_sample(context, response)
    all_ids.append(ids)
    all_masks.append(mask)

#converts the lists into tensors
all_ids = torch.tensor(all_ids, dtype=torch.long)
all_masks = torch.tensor(all_masks, dtype=torch.long)

# splits the data into train and test based on the first 90% of pairs. 
# note: this split is chronological, not random
n = int(0.9 * len(all_ids))
train_ids, test_ids = all_ids[:n], all_ids[n:]
train_masks, test_masks = all_masks[:n], all_masks[n:]

# gets batches
def get_batch(ids, masks):
    # picks batch_size random sample indices
    ix = torch.randint(0, len(ids), (batch_size,))
    # x is all tokens except last one
    x = ids[ix, :-1].to(device)
    # y is all tokens except first one
    y = masks[ix, 1:].to(device)
    return x, y

# checkpoints makes it so you can save the weights of your model so you can pick back up from 
# where you left off while training nistead of fully retraining
os.makedirs('checkpoints_ft', exist_ok=True)
# starts my wandb
wandb.init(project="beta-berra-finetune", config={
    "d_model": d_model, "seq_len": seq_len, "nhead": nhead,
    "n_layers": n_layers, "batch_size": batch_size, "lr": lr,
    "n_steps": n_steps, "context_window": CONTEXT_WINDOW
})
# initializes my model etc
model = Transformer(vocab_size, d_model, seq_len, nhead, n_layers).to(device)
# starts fine-tuning from where the pre-training left off
model.load_state_dict(torch.load('checkpoints/best.pt', map_location=device))

# configuring LoRA for finetuning
config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj", "linear"]
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


# loss function. tells it which things to ignore
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
# weight decay slowly pushes weights towards zero so they don't overfit
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# function that helps us estimate val loss during training
# torch no grad tells the model to not update/compute gradients based on this
@torch.no_grad()
# you give the number of batches to average loss over
def estimate_loss(n_batches=20):
    # switch to evaluation mode - disables dropout
    model.eval()
    results = {}
    # loops twice, once for train and once for test data
    for split, ids, masks in [('train', train_ids, train_masks), ('val', test_ids, test_masks)]:
        losses = []
        # for each batch, calculate the model's loss and add it to running losses
        for _ in range(n_batches):
            x, y = get_batch(ids, masks)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
        # averages all losses across the split (train or test)
        results[split] = sum(losses) / len(losses)
    # puts model back in training mode
    model.train()
    return results

best_val_loss = float('inf')

# ACTUAL TRAINING - do it for n_steps (epochs) ------------------------------------------------------------------------------------------------------------

for step in range(n_steps):
    # gives a random batch of input and output sequences 
    x, y = get_batch(train_ids, train_masks)

    # forward pass - runs x through the model to get predictions
    logits = model(x)
    # calculates loss from logits given a loss function. .view() reshapes x and y
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    # clears gradients since python usually accumulates
    optimizer.zero_grad()
    # backpropagation, computes new gradients
    loss.backward()
    # clips them to have a max of 1 to prevent gradient exploding. prevents training unstability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # updates model's weights based on the gradients
    optimizer.step()

    # every something steps, evaluate losses and print them
    if step % eval_every == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "step": step})
        torch.save(model.state_dict(), f'checkpoints_ft/step_{step}.pt')
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'checkpoints_ft/best.pt')



# end wandb
wandb.finish()