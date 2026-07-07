# imports
import torch
from torch import nn
from datasets import load_from_disk
from tokenizers import Tokenizer
import wandb
from model import Transformer
import os
import math

# for reproducability
torch.manual_seed(873947)

# use gpu instead of cpu
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# using gpt2 tokenizer which uses byte level encoding and byte pair encoding
tokenizer = Tokenizer.from_pretrained("gpt2")
# knowing vocab size is important for the model. tokenizer has get_vocab_size for it.
vocab_size = tokenizer.get_vocab_size()

# loading previously cleaned data saved onto our disk
train_ds = load_from_disk("cleaned_dataset_train")
test_ds = load_from_disk("cleaned_dataset_test")

# tokenizer has a function called encode. this is like my stoi dictionary. concerts to bytes. 
# in our data, every line has a "text" key that we use to get the actual data. 
# this is getting the data, encoding it to get ids, then returning a dictionary with the ids
def tokenize(example):
    ids = tokenizer.encode(example["text"]).ids
    return {"ids": ids}

# tokenizing everything in our train and test datasets. mapping is like applying a function to everything
train_ds = train_ds.map(tokenize)
test_ds = test_ds.map(tokenize)

# makes a function that flattens all ids into a single list instead of a dictionary
def get_flat_ids(ds):
    all_ids = []
    for example in ds:
        all_ids.extend(example["ids"])
    return torch.tensor(all_ids, dtype=torch.long)

# applies the function on all train and test data
train_data = get_flat_ids(train_ds).to(device)
test_data = get_flat_ids(test_ds).to(device)

# hyperparameters for depth and size of model etc
d_model = 256
seq_len = 128
nhead = 4
n_layers = 6
batch_size = 32
lr = 3e-4
n_steps = 5000

# i'm using batching for my training. this function gets batches
def get_batch(data):
    # picks 32 random indices that will be our different batches. 
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    # gets all inputs of length seq_len from each of the indices.
    x = torch.stack([data[i:i+seq_len] for i in ix])
    # gets all outputs from the indices (inputs shifted over 1)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# checkpoints makes it so you can save the weights of your model so you can pick back up from 
# where you left off while training nistead of fully retraining
os.makedirs('checkpoints', exist_ok=True)

# starts my wandb
wandb.init(project="beta-berra", config={
    "d_model": d_model, "seq_len": seq_len, "nhead": nhead,
    "n_layers": n_layers, "batch_size": batch_size, "lr": lr, "n_steps": n_steps
})

# initializes my model etc
model   = Transformer(vocab_size, d_model, seq_len, nhead, n_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
# weight decay slowly pushes weights towards zero so they don't overfit
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# warms up the lr scheduler before we start training with it which is standard
# not too sure on the computations
warmup_steps = 200
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    # cosine decay after warmup
    progress = (step - warmup_steps) / (n_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
# scheduler adjusts learning rate so it finds the optimal weights. lr goes up then comes down to find loss minima
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# function that helps us estimate val loss during training
# torch no grad tells the model to not update/compute gradients based on this
@torch.no_grad()
# you give the number of batches to average loss over
def estimate_loss(n_batches=20):
    # switch to evaluation mode - disables dropout
    model.eval()
    results = {}
    # loops twice, once for train and once for test data
    for split, data in [('train', train_data), ('val', test_data)]:
        losses = []
        # for each batch, calculate the model's loss and add it to running losses
        for _ in range(n_batches):
            x, y = get_batch(data)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
        # averages all losses across the split (train or test)
        results[split] = sum(losses) / len(losses)
    # puts model back in training mode
    model.train()
    return results

best_val_loss = float('inf')

# ACTUAL TRAINING - do it for n_steps (epochs) -------------------------------------------------------------------------------------------------------
for step in range(n_steps):
    # gives a random batch of input and output sequences 
    x, y = get_batch(train_data)

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
    # updates scheduler to decrease the learning rate.
    scheduler.step()

    # every 200 steps, evaluate losses and print them
    if step % 200 == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "step": step})
        torch.save(model.state_dict(), f'checkpoints/step_{step}.pt')
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'checkpoints/best.pt')

# end wandb
wandb.finish()