# imports
import torch
from tokenizers import Tokenizer
from model import Transformer

# use gpu instead of cpu
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# load tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.get_vocab_size()

# hyperparameters (same as model)
d_model = 256
seq_len = 128
nhead = 4
n_layers = 6

# set up model and load all the correct weights from the end of fine-tuning
model = Transformer(vocab_size, d_model, seq_len, nhead, n_layers).to(device)
model.load_state_dict(torch.load('checkpoints_ft/merged_best.pt', map_location=device))
model.eval()

# turns off gradient computation
@torch.no_grad()
def generate(prompt, max_new_tokens=20, temperature=0.7, top_p=0.9, repetition_penalty=1.3):
    # converts starting text into token ids
    context = tokenizer.encode(prompt).ids

    # loop that generates one new token per iteration
    for _ in range(max_new_tokens):
        # grabs the most recent seq_len tokens
        input_seq = context[-seq_len:]
        # keeps track of real_len before padding
        real_len = len(input_seq)
        # if it's too short, then right pads
        if len(input_seq) < seq_len:
            input_seq = input_seq + [0] * (seq_len - real_len)
        
        # turns token list into a tensor then adds a batch dimension with unsqueeze
        x = torch.tensor(input_seq, device=device).unsqueeze(0)
        # forward pass of model + reshaping
        logits = model(x)[0, real_len - 1, :]

        # repetition penalty - if a token has appeared in the context already, 
        # it's score is shrunk so it is less likely to be picked again
        for tok in set(input_seq[:real_len]):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty

        # temperature - divides all scores by a number less than 1 which sharpens the distribution
        logits = logits / temperature

        # top-p (nucleus) filtering
        # arranges scores/tokens from most to least likely
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        # softmax as always to convert scores into probabilities
        probs = torch.softmax(sorted_logits, dim=-1)
        # goes through list and record the cumulative probabilities
        cum_probs = torch.cumsum(probs, dim=-1)
        # once we see %top_p of the data, then does a cutoff to ignore small likelihoods
        cutoff = cum_probs > top_p
        # shifts by one
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        # makes logits of values we want -infinity so they get ignored
        sorted_logits[cutoff] = float('-inf')

        # recalculates probabilities with softmax on the logits with the cutoff applied
        final_probs = torch.zeros_like(logits)
        final_probs[sorted_idx] = torch.softmax(sorted_logits, dim=-1)

        # samples from this probability distribution
        next_token = torch.multinomial(final_probs, num_samples=1).item()
        # adds the predicted token to the context (autoregressive generation)
        context.append(next_token)

        # stop when a text would naturally end
        if next_token == tokenizer.encode('\n').ids[0]:
            break
    
    # return entire predicted text
    return tokenizer.decode(context)

print(generate("THEM: hey how are you\nME:"))
